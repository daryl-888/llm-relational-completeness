"""Phase 2: JOIN — list every matching pair across two tables."""
from __future__ import annotations

from ..schema import Question
from ..synth import badges, customers, employees, orders, projects
from .common import Ctx, condition, entity_universe, fmt_for, sample_query, sort_rows


def _one_to_many(ctx: Ctx, qid: str, i: int):
    n_c, n_o = [(30, 100), (60, 200), (120, 400)][i % 3]
    c = ctx.table(customers(f"{qid}.customers", ctx.rng, ctx.nm, n_c))
    o = ctx.table(orders(f"{qid}.orders", ctx.rng, ctx.nm, n_o, c.col("name")))

    def propose():
        cc = condition(ctx.rng, c, ctx.rng.choice(["city", "tier", "signup_year"]), qualify=True)
        oc = condition(ctx.rng, o, ctx.rng.choice(["category", "amount", "year"]), qualify=True)
        sql = (f"SELECT customers.name, orders.order_id FROM customers JOIN orders ON orders.customer = "
               f"customers.name WHERE {cc.sql} AND {oc.sql}")
        return sql, [c, o], (cc, oc)

    rows, sql, (cc, oc) = sample_query(ctx.rng, propose, 3, 60)
    text = (f"List every (customer name, order ID) pair such that the order was placed by that customer, "
            f"{cc.of('customer')}, and {oc.of('order')}.")
    return [c, o], rows, sql, text, ["customer name", "order ID"], [entity_universe(c), entity_universe(o)], \
        {"n_rows": n_c + n_o}


def _many_to_many(ctx: Ctx, qid: str, i: int):
    n_e, n_p = [(40, 10), (80, 20), (150, 40)][i % 3]
    cities = ctx.nm.cities(6)
    e = ctx.table(employees(f"{qid}.employees", ctx.rng, ctx.nm, n_e, cities=cities))
    p = ctx.table(projects(f"{qid}.projects", ctx.rng, ctx.nm, n_p, cities))

    def propose():
        ec = condition(ctx.rng, e, ctx.rng.choice(["department", "role", "remote", "years_experience"]), qualify=True)
        pc = condition(ctx.rng, p, ctx.rng.choice(["status", "budget"]), qualify=True)
        sql = (f"SELECT employees.name, projects.name FROM employees JOIN projects ON employees.city = "
               f"projects.city WHERE {ec.sql} AND {pc.sql}")
        return sql, [e, p], (ec, pc)

    rows, sql, (ec, pc) = sample_query(ctx.rng, propose, 3, 60)
    text = (f"List every (employee name, project name) pair such that the employee's office city is the same as "
            f"the project's city, {ec.of('employee')}, and {pc.of('project')}.")
    return [e, p], rows, sql, text, ["employee name", "project name"], [entity_universe(e), entity_universe(p)], \
        {"n_rows": n_e + n_p}


def _lookup(ctx: Ctx, qid: str, i: int):
    n = [50, 100, 200][i % 3]
    e = ctx.table(employees(f"{qid}.employees", ctx.rng, ctx.nm, n))
    b = ctx.table(badges(f"{qid}.badges", ctx.rng, ctx.nm, e.col("name")))

    def propose():
        ec = condition(ctx.rng, e, ctx.rng.choice(["department", "role", "city", "salary"]), qualify=True)
        bc = condition(ctx.rng, b, ctx.rng.choice(["access_level", "building"]), qualify=True)
        sql = (f"SELECT employees.name, badges.badge_id FROM employees JOIN badges ON badges.employee = "
               f"employees.name WHERE {ec.sql} AND {bc.sql}")
        return sql, [e, b], (ec, bc)

    rows, sql, (ec, bc) = sample_query(ctx.rng, propose, 3, 40)
    text = (f"List every (employee name, badge ID) pair such that the badge belongs to that employee, "
            f"{ec.of('employee')}, and {bc.of('badge')}.")
    return [e, b], rows, sql, text, ["employee name", "badge ID"], [entity_universe(e), entity_universe(b)], \
        {"n_rows": 2 * n}


KINDS = [("1:N", _one_to_many), ("M:N", _many_to_many), ("1:1", _lookup)]


def join(ctx: Ctx, count: int = 60) -> None:
    for i in range(count):
        sub, make = KINDS[i % 3]
        qid = f"join-{i:03d}"
        tables, rows, sql, text, fields, universe, meta = make(ctx, qid, i // 3)
        ctx.add(Question(
            id=qid, op="join", setting="context", subtype=sub, answer_type="set", fields=fields, question=text,
            fmt=fmt_for(i // 3), tables=[t.id for t in tables], sql=sql, gold=sort_rows(rows), universe=universe,
            meta={"n_gold": len(rows), **meta}))
