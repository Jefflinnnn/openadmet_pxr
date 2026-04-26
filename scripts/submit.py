#!/usr/bin/env python
"""Submit predictions to the OpenADMET PXR challenge leaderboard and fetch results."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from openadmet_pxr.submission.validate import validate_submission

HF_SPACE = "https://openadmet-pxr-challenge.hf.space/"
USERNAME = "jefflinnnn"
USER_ALIAS = "HungryCapybara"
PARTICIPANT_NAME = "Jeff Lin"
EMAIL = "Jeff.lin@ucsf.edu"
AFFILIATION = "UCSF"


def fetch_leaderboard() -> pd.DataFrame:
    from gradio_client import Client
    client = Client(HF_SPACE)
    path = client.predict(api_name="/download_activity_leaderboard")
    df = pd.read_csv(path)
    return df


def show_leaderboard(top_n: int = 10, show_ours: bool = True) -> None:
    df = fetch_leaderboard()
    cols = ["rank", "username", "MAE", "RAE", "R2", "Spearman ρ", "Kendall's τ"]
    cols = [c for c in cols if c in df.columns]

    print(f"\n=== Activity Leaderboard (top {top_n}) ===")
    print(df[cols].head(top_n).to_string(index=False))

    if show_ours:
        ours = df[df["username"] == USER_ALIAS]
        if not ours.empty:
            print(f"\n=== {USER_ALIAS} ===")
            print(ours[cols].to_string(index=False))
        print(f"\nTotal entries: {len(df)}")


def submit(csv_path: str, model_tag: str) -> bool:
    """Returns True if submission was accepted, False if on cooldown."""
    from gradio_client import Client, handle_file

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")

    validate_submission(path)

    client = Client(HF_SPACE)
    result = client.predict(
        username=USERNAME,
        user_alias=USER_ALIAS,
        anon_checkbox=True,
        participant_name=PARTICIPANT_NAME,
        discord_username="",
        email=EMAIL,
        affiliation=AFFILIATION,
        model_tag=model_tag,
        paper_checkbox=False,
        proprietary_data_checkbox=False,
        track_select="Activity Prediction",
        file_input=handle_file(str(path.resolve())),
        api_name="/submit_predictions",
    )

    import re
    result_str = str(result)
    result_lower = result_str.lower()
    on_cooldown = any(w in result_lower for w in ("cooldown", "wait", "too soon", "limit", "please wait"))

    print(f"Submitted: {path.name}")
    print(f"Response:  {result_str}")

    if on_cooldown:
        # Extract remaining time if present, e.g. "Please wait 02:50:32 before submitting again"
        match = re.search(r"(\d{2}:\d{2}:\d{2})", result_str)
        wait_str = f" ({match.group(1)} remaining)" if match else ""
        print(f"\n⏳ Cooldown active{wait_str} — submission not accepted.")
        return False

    print("\nFetching updated leaderboard...")
    show_leaderboard(top_n=10)
    return True


def main():
    parser = argparse.ArgumentParser(description="Submit to OpenADMET PXR leaderboard")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit a prediction CSV")
    p_submit.add_argument("csv", help="Path to submission CSV")
    p_submit.add_argument("--tag", default="", help="Model tag / description")

    sub.add_parser("leaderboard", help="Fetch and display current leaderboard")

    args = parser.parse_args()

    if args.command == "submit":
        submit(args.csv, args.tag)
    elif args.command == "leaderboard":
        show_leaderboard(top_n=15)


if __name__ == "__main__":
    main()
