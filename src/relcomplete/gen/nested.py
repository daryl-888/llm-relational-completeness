"""Phase 2: nested queries — IN, NOT IN, correlated and scalar subqueries."""
from __future__ import annotations

from ..schema import Question, lit
from ..synth import CATEGORIES, DEPARTMENTS, ROLES, STATUSES, customers, employees, orders, products, projects
from .common import Ctx, condition, entity_universe, fmt_for, sample_query, sort_rows


def _emp_proj(ctx: Ctx, qid: str, i: int):
    n_e, n_p = [(60, 15), (120, 25), (200, 40)][i % 3]
    cities = ctx.nm.cities(12)
    e = ctx.table(employees(f"{qid}.employees", ctx.rng, ctx.nm, n_e, cities=cities))
    p = ctx.table(projects(f"{qid}.projects", ctx.rng, ctx.nm, n_p, cities))
    return e, p


def _cust_orders(ctx: Ctx, qid: str, i: int):
    n_c, n_o = [(40, 80), (80, 160), (150, 300)][i % 3]
    c = ctx.table(customers(f"{qid}.customers", ctx.rng, ctx.nm, n_c))
    return c, ctx.table(orders(f"{qid}.orders", ctx.rng, ctx.nm, n_o, c.col("name")))


def _in(ctx: Ctx, qid: str, i: int, negate: bool):
    r = ctx.rng
    neg = "NOT " if negate else ""
    if i % 2 == 0:
        e, p = _emp_proj(ctx, qid, i)

        def propose():
            st = r.choice(STATUSES)
            ec = condition(r, e, r.choice(["department", "role", "remote"]))
            sql = (f"SELECT name FROM employees WHERE city {neg}IN (SELECT city FROM projects WHERE status = "
                   f"{lit(st)}) AND {ec.sql}")
            return sql, [e, p], (st, ec)

        rows, sql, (st, ec) = sample_query(r, propose, 3, 50)
        rel = "no project" if negate else "at least one project"
        text = (f"List the names of all employees {ec.whose()} and whose office city has {rel} with status "
                f"{st}.")
        return [e, p], rows, sql, text, "employee name", entity_universe(e)
    c, o = _cust_orders(ctx, qid, i)

    def propose():
        oc = condition(r, o, r.choice(["category", "amount", "year"]))
        cc = condition(r, c, r.choice(["tier", "city"]))
        sql = (f"SELECT name FROM customers WHERE name {neg}IN (SELECT customer FROM orders WHERE {oc.sql}) "
               f"AND {cc.sql}")
        return sql, [c, o], (oc, cc)

    rows, sql, (oc, cc) = sample_query(r, propose, 3, 50)
    verb = "did not place any order" if negate else "placed at least one order"
    text = f"List the names of all customers {cc.whose()} who {verb} {oc.whose()}."
    return [c, o], rows, sql, text, "customer name", entity_universe(c)


def _correlated(ctx: Ctx, qid: str, i: int):
    r = ctx.rng
    if i % 2 == 0:
        e = ctx.table(employees(f"{qid}.employees", r, ctx.nm, [60, 120, 200][i % 3]))

        def propose():
            role = r.choice(ROLES)
            sql = (f"SELECT e.name FROM employees e WHERE e.role = {lit(role)} AND e.salary > (SELECT AVG(salary) "
                   f"FROM employees d WHERE d.department = e.department)")
            return sql, [e], role

        rows, sql, role = sample_query(r, propose, 3, 50)
        text = (f"List the names of all employees whose role is {role} and whose annual salary is higher than the "
                f"average annual salary of all employees (any role) in their own department.")
        return [e], rows, sql, text, "employee name", entity_universe(e)
    pr = ctx.table(products(f"{qid}.products", r, ctx.nm, [60, 120, 200][i % 3]))

    def propose():
        brand = r.choice(pr.col("brand"))
        sql = (f"SELECT p.name FROM products p WHERE p.brand = {lit(brand)} AND p.price > (SELECT AVG(price) "
               f"FROM products q WHERE q.category = p.category)")
        return sql, [pr], brand

    rows, sql, brand = sample_query(r, propose, 3, 50)
    text = (f"List the names of all products made by {brand} whose price is higher than the average price of all "
            f"products (any brand) in the same category.")
    return [pr], rows, sql, text, "product name", entity_universe(pr)


def _scalar(ctx: Ctx, qid: str, i: int):
    r = ctx.rng
    if i % 2 == 0:
        pr = ctx.table(products(f"{qid}.products", r, ctx.nm, [60, 120, 200][i % 3]))

        def propose():
            cat = r.choice(CATEGORIES)
            sql = f"SELECT name FROM products WHERE price > (SELECT MAX(price) FROM products WHERE category = {lit(cat)})"
            return sql, [pr], cat

        rows, sql, cat = sample_query(r, propose, 3, 50)
        text = f"List the names of all products that cost more than every product in the {cat} category."
        return [pr], rows, sql, text, "product name", entity_universe(pr)
    e = ctx.table(employees(f"{qid}.employees", r, ctx.nm, [60, 120, 200][i % 3]))

    def propose():
        dept = r.choice(DEPARTMENTS)
        sql = (f"SELECT name FROM employees WHERE years_experience < (SELECT MIN(years_experience) FROM employees "
               f"WHERE department = {lit(dept)})")
        return sql, [e], dept

    rows, sql, dept = sample_query(r, propose, 3, 50)
    text = (f"List the names of all employees who have fewer years of experience than every employee in the "
            f"{dept} department.")
    return [e], rows, sql, text, "employee name", entity_universe(e)


PLAN = [("in", lambda c, q, i: _in(c, q, i, False)), ("not_in", lambda c, q, i: _in(c, q, i, True)),
        ("correlated", _correlated), ("scalar", _scalar)]


def nested(ctx: Ctx, count: int = 40) -> None:
    for i in range(count):
        sub, make = PLAN[i % 4]
        qid = f"nest-{i:03d}"
        tables, rows, sql, text, field, universe = make(ctx, qid, i // 4)
        ctx.add(Question(
            id=qid, op="nested", setting="context", subtype=sub, answer_type="set", fields=[field], question=text,
            fmt=fmt_for(i // 4), tables=[t.id for t in tables], sql=sql, gold=sort_rows(rows), universe=[universe],
            meta={"n_gold": len(rows), "n_rows": sum(len(t.rows) for t in tables)}))
