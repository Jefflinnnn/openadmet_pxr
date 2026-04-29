#!/usr/bin/env python
"""Submit predictions to the OpenADMET PXR challenge leaderboard and fetch results."""

import argparse
import re
import sys
import time
from datetime import datetime, timezone
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

COOLDOWN_HOURS = 4
BUFFER_SECONDS = 5 * 60  # 5 minute buffer after cooldown lifts


def fetch_leaderboard() -> pd.DataFrame:
    from gradio_client import Client
    client = Client(HF_SPACE)
    path = client.predict(api_name="/download_activity_leaderboard")
    return pd.read_csv(path)


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


def _parse_cooldown_seconds(result_str: str) -> int | None:
    """Extract remaining wait time in seconds from a cooldown response."""
    match = re.search(r"(\d{2}):(\d{2}):(\d{2})", result_str)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return h * 3600 + m * 60 + s
    return None


def _is_cooldown(result_str: str) -> bool:
    return any(w in result_str.lower() for w in ("cooldown", "please wait", "too soon", "limit"))


def submit(csv_path: str, model_tag: str) -> bool:
    """Attempt submission. Returns True if accepted, False if on cooldown."""
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

    result_str = str(result)
    print(f"Submitted: {path.name}")
    print(f"Response:  {result_str}")

    if _is_cooldown(result_str):
        secs = _parse_cooldown_seconds(result_str)
        wait_str = f" ({secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d} remaining)" if secs else ""
        print(f"\n⏳ Cooldown active{wait_str} — submission not accepted.")
        return False

    print("\nFetching updated leaderboard...")
    show_leaderboard(top_n=10)
    return True


def submit_when_ready(csv_path: str, model_tag: str) -> None:
    """Attempt submission; if on cooldown, wait out the remaining time + buffer then retry."""
    # First attempt
    from gradio_client import Client, handle_file

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    validate_submission(path)

    while True:
        accepted = submit(csv_path, model_tag)
        if accepted:
            return

        # Re-probe to get fresh remaining time
        from gradio_client import Client, handle_file
        client = Client(HF_SPACE)
        result = str(client.predict(
            username=USERNAME, user_alias=USER_ALIAS, anon_checkbox=True,
            participant_name=PARTICIPANT_NAME, discord_username="", email=EMAIL,
            affiliation=AFFILIATION, model_tag=model_tag, paper_checkbox=False,
            proprietary_data_checkbox=False, track_select="Activity Prediction",
            file_input=handle_file(str(path.resolve())),
            api_name="/submit_predictions",
        ))

        wait_secs = _parse_cooldown_seconds(result)
        if wait_secs is None:
            wait_secs = COOLDOWN_HOURS * 3600  # fallback: wait full cooldown

        total_wait = wait_secs + BUFFER_SECONDS
        ready_at = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        print(f"\n⏳ Cooldown: {wait_secs//3600:02d}h{(wait_secs%3600)//60:02d}m remaining "
              f"+ 5min buffer = waiting {total_wait//60:.0f} min total.")
        print(f"   Will submit at ~{ready_at} + {total_wait//60:.0f}min")

        time.sleep(total_wait)
        print(f"\n⏰ Cooldown elapsed, attempting submission...")


def main():
    parser = argparse.ArgumentParser(description="Submit to OpenADMET PXR leaderboard")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit immediately (reports cooldown if blocked)")
    p_submit.add_argument("csv", help="Path to submission CSV")
    p_submit.add_argument("--tag", default="", help="Model tag / description")

    p_wait = sub.add_parser("submit-when-ready",
                             help="Submit now or wait out cooldown + 5min buffer then submit")
    p_wait.add_argument("csv", help="Path to submission CSV")
    p_wait.add_argument("--tag", default="", help="Model tag / description")

    sub.add_parser("leaderboard", help="Fetch and display current leaderboard")

    args = parser.parse_args()

    if args.command == "submit":
        submit(args.csv, args.tag)
    elif args.command == "submit-when-ready":
        submit_when_ready(args.csv, args.tag)
    elif args.command == "leaderboard":
        show_leaderboard(top_n=15)


if __name__ == "__main__":
    main()
