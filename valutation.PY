from pathlib import Path

import pandas as pd


def compute_valuation(predicted: float, unit_id: int, cycle: int, data_dir: str = "Data"):
	cols = ["unit", "cycle"] + [f"f{i}" for i in range(1, 25)]

	data_path = Path(data_dir)
	test_df = pd.read_csv(data_path / "test_FD001.txt", sep=r"\s+", header=None, engine="python").iloc[:, :26]
	test_df.columns = cols

	rul_df = pd.read_csv(data_path / "RUL_FD001.txt", sep=r"\s+", header=None, engine="python")

	unit_id = int(unit_id)
	cycle = int(cycle)
	predicted = float(predicted)

	rul_end = int(rul_df.iloc[unit_id - 1, 0])
	max_cycle = int(test_df.loc[test_df["unit"] == unit_id, "cycle"].max())
	true_rul = (max_cycle - cycle) + rul_end

	abs_error = abs(predicted - true_rul)
	pct_error = (abs_error / true_rul) * 100 if true_rul != 0 else 0.0

	return {
		"predicted_rul": predicted,
		"true_rul": float(true_rul),
		"absolute_error": float(abs_error),
		"percentage_error": float(pct_error),
	}


if __name__ == "__main__":
	predicted = 123.93612670898438
	unit_id = 1
	cycle = 12

	result = compute_valuation(predicted, unit_id, cycle)
	print("predicted_rul:", result["predicted_rul"])
	print("true_rul:", result["true_rul"])
	print("absolute_error:", result["absolute_error"])
	print("percentage_error:", result["percentage_error"])