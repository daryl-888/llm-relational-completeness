"""Seeded generators for fictional relational tables.

Every entity (people, products, books, projects, cities) is invented so the
model cannot answer from memorised world knowledge: the only way to get a
complete answer is to read every record it was given.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from faker import Faker


@dataclass
class Table:
    id: str
    kind: str  # SQL table name and rendering template, e.g. "employees"
    columns: list[str]
    rows: list[list]
    key: str  # column that identifies an entity

    def col(self, name: str) -> list:
        i = self.columns.index(name)
        return [r[i] for r in self.rows]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Table":
        return cls(**d)


# Column metadata used to phrase conditions in English. type: cat | num | bool
COLUMNS: dict[str, dict[str, tuple[str, str]]] = {
    "employees": {
        "department": ("department", "cat"),
        "role": ("role", "cat"),
        "city": ("office city", "cat"),
        "salary": ("annual salary (USD)", "num"),
        "years_experience": ("years of experience", "num"),
        "hire_year": ("hire year", "num"),
        "remote": ("remote-work status", "bool"),
    },
    "products": {
        "category": ("category", "cat"),
        "brand": ("brand", "cat"),
        "price": ("price (USD)", "num"),
        "rating": ("customer rating", "num"),
        "stock": ("units in stock", "num"),
        "release_year": ("release year", "num"),
    },
    "students": {
        "major": ("major", "cat"),
        "class_year": ("class year", "cat"),
        "hometown": ("hometown", "cat"),
        "gpa": ("GPA", "num"),
        "credits": ("credits earned", "num"),
        "scholarship": ("scholarship status", "bool"),
    },
    "books": {
        "author": ("author", "cat"),
        "genre": ("genre", "cat"),
        "publisher": ("publisher", "cat"),
        "pages": ("page count", "num"),
        "year": ("publication year", "num"),
    },
    "customers": {
        "city": ("city", "cat"),
        "tier": ("membership tier", "cat"),
        "signup_year": ("signup year", "num"),
    },
    "orders": {
        "category": ("product category", "cat"),
        "amount": ("amount (USD)", "num"),
        "year": ("year", "num"),
    },
    "projects": {
        "city": ("city", "cat"),
        "status": ("status", "cat"),
        "budget": ("budget (USD)", "num"),
    },
    "badges": {
        "access_level": ("access level", "cat"),
        "building": ("building", "cat"),
    },
}

# What each kind's key column is called in questions, singular and plural.
ENTITY = {
    "employees": ("employee", "employees", "name"),
    "products": ("product", "products", "name"),
    "students": ("student", "students", "name"),
    "books": ("book", "books", "title"),
    "customers": ("customer", "customers", "name"),
    "orders": ("order", "orders", "order ID"),
    "projects": ("project", "projects", "name"),
    "badges": ("badge", "badges", "badge ID"),
}

DEPARTMENTS = ["Engineering", "Marketing", "Sales", "Finance", "Legal", "Operations", "Support", "Research"]
ROLES = ["Analyst", "Engineer", "Manager", "Specialist", "Coordinator", "Director"]
CATEGORIES = ["Kitchen", "Garden", "Audio", "Office", "Outdoor", "Fitness", "Lighting", "Storage", "Toys", "Tools"]
MAJORS = ["Biology", "Chemistry", "Economics", "History", "Mathematics", "Philosophy", "Physics",
          "Psychology", "Sociology", "Linguistics"]
CLASS_YEARS = ["Freshman", "Sophomore", "Junior", "Senior"]
GENRES = ["Mystery", "Romance", "Fantasy", "Thriller", "Biography", "Poetry", "Horror", "Travel"]
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
STATUSES = ["Active", "Paused", "Completed", "Proposed"]
ACCESS = ["Visitor", "Standard", "Restricted", "Full"]

_SYL = ["ka", "lo", "vex", "dra", "mi", "tor", "zen", "qua", "ri", "sol", "bel", "nor", "tha", "vi",
        "ra", "quin", "dor", "phe", "lys", "mar", "ost", "ul", "fen", "gri", "pax", "or", "ty", "sen"]
_ADJ = ["Silent", "Hollow", "Crimson", "Distant", "Broken", "Golden", "Hidden", "Last", "Quiet", "Burning",
        "Frozen", "Wandering", "Secret", "Pale", "Endless", "Gentle", "Iron", "Velvet", "Bitter", "Lonely"]
_NOUN = ["Harbor", "Lantern", "Orchard", "Archive", "Meridian", "Tide", "Cartographer", "Garden", "Signal",
         "Compass", "Winter", "Ember", "Monument", "Atlas", "Sparrow", "Threshold", "Voyage", "Mirror",
         "Fortress", "Hourglass"]


class Namer:
    """Deterministic source of unique fictional names."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.fk = Faker("en_US")
        self.fk.seed_instance(seed)

    def _unique(self, make, n: int) -> list[str]:
        out, seen = [], set()
        while len(out) < n:
            v = make()
            if v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
        return out

    def people(self, n: int) -> list[str]:
        return self._unique(lambda: f"{self.fk.first_name()} {self.fk.last_name()}", n)

    def cities(self, n: int) -> list[str]:
        return self._unique(self.fk.city, n)

    def word(self) -> str:
        return "".join(self.rng.choice(_SYL) for _ in range(self.rng.choice([2, 3]))).capitalize()

    def words(self, n: int) -> list[str]:
        return self._unique(self.word, n)

    def products(self, n: int) -> list[str]:
        r = self.rng
        return self._unique(lambda: f"{self.word()} {r.choice('ABCDEFGHJKLMNPRSTVXZ')}-{r.randint(100, 999)}", n)

    def titles(self, n: int) -> list[str]:
        r = self.rng
        forms = [lambda: f"The {r.choice(_ADJ)} {r.choice(_NOUN)}",
                 lambda: f"{r.choice(_NOUN)} of {self.word()}",
                 lambda: f"The {r.choice(_NOUN)} of {r.choice(_ADJ)} {self.word()}"]
        return self._unique(lambda: r.choice(forms)(), n)

    def codes(self, prefix: str, n: int) -> list[str]:
        return [f"{prefix}-{x}" for x in self.rng.sample(range(1000, 10000), n)]


def _shuffled(rng: random.Random, rows: list[list]) -> list[list]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows


def employees(tid: str, rng: random.Random, nm: Namer, n: int, cities: list[str] | None = None,
              names: list[str] | None = None) -> Table:
    cities = cities or nm.cities(10)
    names = names or nm.people(n)
    rows = []
    for name in names:
        years = rng.randint(0, 30)
        rows.append([name, rng.choice(DEPARTMENTS), rng.choice(ROLES), rng.choice(cities),
                     rng.randrange(40_000, 200_001, 1_000), years, rng.randint(2005, 2024),
                     rng.choice(["yes", "no"])])
    cols = ["name", "department", "role", "city", "salary", "years_experience", "hire_year", "remote"]
    return Table(tid, "employees", cols, rows, "name")


def products(tid: str, rng: random.Random, nm: Namer, n: int) -> Table:
    brands = nm.words(12)
    rows = [[name, rng.choice(CATEGORIES), rng.choice(brands), rng.randint(5, 2000),
             round(rng.uniform(1.0, 5.0), 1), rng.randint(0, 500), rng.randint(2015, 2024)]
            for name in nm.products(n)]
    cols = ["name", "category", "brand", "price", "rating", "stock", "release_year"]
    return Table(tid, "products", cols, rows, "name")


def students(tid: str, rng: random.Random, nm: Namer, n: int, names: list[str] | None = None) -> Table:
    towns = nm.cities(10)
    rows = [[name, rng.choice(MAJORS), rng.choice(CLASS_YEARS), rng.choice(towns),
             round(rng.uniform(2.0, 4.0), 2), rng.randint(0, 130), rng.choice(["yes", "no"])]
            for name in (names or nm.people(n))]
    cols = ["name", "major", "class_year", "hometown", "gpa", "credits", "scholarship"]
    return Table(tid, "students", cols, rows, "name")


def books(tid: str, rng: random.Random, nm: Namer, n: int) -> Table:
    authors = nm.people(30)
    publishers = [f"{w} Press" for w in nm.words(8)]
    rows = [[t, rng.choice(authors), rng.choice(GENRES), rng.choice(publishers), rng.randint(90, 900),
             rng.randint(1960, 2024)] for t in nm.titles(n)]
    cols = ["title", "author", "genre", "publisher", "pages", "year"]
    return Table(tid, "books", cols, rows, "title")


def customers(tid: str, rng: random.Random, nm: Namer, n: int) -> Table:
    cities = nm.cities(8)
    rows = [[name, rng.choice(cities), rng.choice(TIERS), rng.randint(2015, 2024)] for name in nm.people(n)]
    return Table(tid, "customers", ["name", "city", "tier", "signup_year"], rows, "name")


def orders(tid: str, rng: random.Random, nm: Namer, n: int, customer_names: list[str]) -> Table:
    rows = [[oid, rng.choice(customer_names), rng.choice(CATEGORIES), rng.randint(10, 1500), rng.randint(2021, 2024)]
            for oid in nm.codes("ORD", n)]
    return Table(tid, "orders", ["order_id", "customer", "category", "amount", "year"], rows, "order_id")


def projects(tid: str, rng: random.Random, nm: Namer, n: int, cities: list[str]) -> Table:
    rows = [[f"Project {w}", rng.choice(cities), rng.choice(STATUSES), rng.randrange(50_000, 2_000_001, 10_000)]
            for w in nm.words(n)]
    return Table(tid, "projects", ["name", "city", "status", "budget"], rows, "name")


def badges(tid: str, rng: random.Random, nm: Namer, employee_names: list[str]) -> Table:
    buildings = [f"Building {c}" for c in "ABCDE"]
    ids = nm.codes("BDG", len(employee_names))
    rows = [[bid, emp, rng.choice(ACCESS), rng.choice(buildings)] for bid, emp in zip(ids, employee_names)]
    return Table(tid, "badges", ["badge_id", "employee", "access_level", "building"], _shuffled(rng, rows),
                 "badge_id")


def roster(tid: str, kind: str, rng: random.Random, members: list[str], class_year: dict[str, str]) -> Table:
    rows = [[m, class_year[m]] for m in members]
    return Table(tid, kind, ["name", "class_year"], _shuffled(rng, rows), "name")


GENERATORS = {"employees": employees, "products": products, "students": students, "books": books}
# The categorical column each enumeration question filters on, per domain.
ENUM_COLUMN = {"employees": ("department", DEPARTMENTS), "products": ("category", CATEGORIES),
               "students": ("major", MAJORS), "books": ("genre", GENRES)}
