"""Собирает дневные продажи CA_1 / FOODS из файлов M5."""

import csv
import hashlib
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


SOURCE = "https://zenodo.org/records/10203108/files"
FILES = {
    "sales_train_evaluation.csv": "b806dfc9f30a745102b708c09951f6aa",
    "calendar.csv": "3ffeab2991b0c8e861d008b39ea4c95c",
}
DAY_COUNT = 1941
FIRST_DATE = date(2011, 1, 29)


def prepare_sales(output_path: Path) -> None:
    """Скачивает исходные файлы и сохраняет дневные продажи в CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent) as directory:
        folder = Path(directory)
        for filename, expected_md5 in FILES.items():
            _download(folder / filename, expected_md5)
        staged = folder / "sales.csv"
        _write_aggregate(folder, staged)
        staged.replace(output_path)


def _download(path: Path, expected_md5: str) -> None:
    checksum = hashlib.md5(usedforsecurity=False)
    with urlopen(f"{SOURCE}/{path.name}?download=1", timeout=60) as response:
        with path.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
                checksum.update(chunk)
    if checksum.hexdigest() != expected_md5:
        raise ValueError(f"Не совпала контрольная сумма файла {path.name}.")


def _write_aggregate(folder: Path, output_path: Path) -> None:
    days = [f"d_{number}" for number in range(1, DAY_COUNT + 1)]
    totals = [0] * DAY_COUNT
    items = set()
    with (folder / "sales_train_evaluation.csv").open(newline="", encoding="utf-8") as file:
        sales = csv.reader(file)
        header = next(sales)
        metadata = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        if header != metadata + days:
            raise ValueError("Столбцы в файле продаж M5 не совпадают с ожидаемыми.")
        for row in sales:
            if row[3] != "FOODS" or row[4] != "CA_1":
                continue
            if row[1] in items:
                raise ValueError(f"Товар {row[1]} повторяется в CA_1.")
            items.add(row[1])
            for index, units in enumerate(row[6:]):
                totals[index] += int(units)
    if not items:
        raise ValueError("В данных нет продаж CA_1 / FOODS.")

    with (folder / "calendar.csv").open(newline="", encoding="utf-8") as file:
        calendar = {row["d"]: row for row in csv.DictReader(file)}
    rows = []
    for index, day in enumerate(days):
        calendar_day = calendar[day]
        current_date = date.fromisoformat(calendar_day["date"])
        if current_date != FIRST_DATE + timedelta(days=index):
            raise ValueError(f"В календаре указана неверная дата для {day}.")
        is_event = bool(calendar_day["event_name_1"] or calendar_day["event_name_2"])
        rows.append((current_date.isoformat(), totals[index], int(is_event)))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "units", "is_event"])
        writer.writerows(rows)


if __name__ == "__main__":
    prepare_sales(Path("data/sales.csv"))
