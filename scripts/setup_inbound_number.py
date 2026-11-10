"""Connect the Talkbalt LiveKit Phone Number to the agent (one-time, idempotent).

Creates a callee dispatch rule that puts each inbound call in its own room
(`call-<dialed number>_<random>`) and explicitly dispatches `talkbalt-agent`, then assigns
the rule to the phone number. A callee rule is used instead of an individual rule because
an individual rule can name the room after the caller's phone number, and room names are
logged by LiveKit and never redacted.

Usage (from backend-web):
    uv run scripts/setup_inbound_number.py           # read-only: show what would happen
    uv run scripts/setup_inbound_number.py --apply   # create and assign the rule

If the number already has a dispatch rule, the script prints it and changes nothing.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from livekit import api

PHONE_NUMBER = "+12678019838"
AGENT_NAME = "talkbalt-agent"
RULE_NAME = "talkbalt-inbound"
ROOM_PREFIX = "call-"

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")


class PhoneNumbers:
    """Minimal client for the Phone Numbers Twirp API, which the Python SDK doesn't wrap yet.

    See https://docs.livekit.io/reference/telephony/phone-numbers-api/
    """

    def __init__(self, session: aiohttp.ClientSession) -> None:
        url = os.environ["LIVEKIT_URL"]
        self._base = url.replace("wss://", "https://").replace("ws://", "http://").rstrip("/")
        self._session = session
        self._token = (
            api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_sip_grants(api.SIPGrants(admin=True))
            .to_jwt()
        )

    async def _call(self, method: str, body: dict) -> dict:
        async with self._session.post(
            f"{self._base}/twirp/livekit.PhoneNumberService/{method}",
            json=body,
            headers={"Authorization": f"Bearer {self._token}"},
        ) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                raise RuntimeError(f"{method} failed ({resp.status}): {data}")
            return data

    async def get(self, phone_number: str) -> dict:
        return (await self._call("GetPhoneNumber", {"phone_number": phone_number}))["phone_number"]

    async def assign_rule(self, phone_number: str, rule_id: str) -> dict:
        body = {"phone_number": phone_number, "sip_dispatch_rule_id": rule_id}
        return (await self._call("UpdatePhoneNumber", body))["phone_number"]


def rule_ids_of(number: dict) -> list[str]:
    # Twirp JSON may use proto or camelCase names; the singular field is deprecated.
    ids = number.get("sip_dispatch_rule_ids") or number.get("sipDispatchRuleIds") or []
    single = number.get("sip_dispatch_rule_id") or number.get("sipDispatchRuleId")
    return list(dict.fromkeys([*ids, *([single] if single else [])]))


def describe_number(number: dict) -> str:
    return (
        f"{number.get('e164_format') or number.get('e164Format')} "
        f"id={number.get('id')} status={number.get('status')} rules={rule_ids_of(number) or 'none'}"
    )


def describe_rule(rule: api.SIPDispatchRuleInfo) -> str:
    kind = rule.rule.WhichOneof("rule")
    agents = [a.agent_name for a in rule.room_config.agents]
    return f"{rule.sip_dispatch_rule_id} name={rule.name!r} type={kind} agents={agents}"


async def main(apply: bool) -> None:
    lkapi = api.LiveKitAPI()
    try:
        async with aiohttp.ClientSession() as http:
            numbers = PhoneNumbers(http)
            number = await numbers.get(PHONE_NUMBER)
            print("phone number:", describe_number(number))

            rules = (await lkapi.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())).items
            by_id = {r.sip_dispatch_rule_id: r for r in rules}

            assigned = rule_ids_of(number)
            if assigned:
                print("This number already has a dispatch rule. Nothing to do.")
                for rule_id in assigned:
                    rule = by_id.get(rule_id)
                    print("  rule:", describe_rule(rule) if rule else f"{rule_id} (details not found)")
                return

            # Reuse our rule if an earlier run created it but didn't get to assign it.
            existing = next((r for r in rules if r.name == RULE_NAME), None)
            if not apply:
                action = f"reuse {describe_rule(existing)}" if existing else f"create rule {RULE_NAME!r}"
                print(f"Read-only run. With --apply this would {action} and assign it to {PHONE_NUMBER}.")
                return

            if existing:
                rule = existing
                print("reusing rule:", describe_rule(rule))
            else:
                rule = await lkapi.sip.create_dispatch_rule(
                    api.CreateSIPDispatchRuleRequest(
                        dispatch_rule=api.SIPDispatchRuleInfo(
                            name=RULE_NAME,
                            rule=api.SIPDispatchRule(
                                dispatch_rule_callee=api.SIPDispatchRuleCallee(
                                    room_prefix=ROOM_PREFIX,
                                    randomize=True,  # a separate room for every call
                                )
                            ),
                            room_config=api.RoomConfiguration(
                                agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)]
                            ),
                        )
                    )
                )
                print("created rule:", describe_rule(rule))

            try:
                number = await numbers.assign_rule(PHONE_NUMBER, rule.sip_dispatch_rule_id)
            except RuntimeError as e:
                # Seen on this project: UpdatePhoneNumber returns a generic 400 even with valid
                # IDs. The rule exists, so finish the assignment in the dashboard.
                print(f"Could not assign the rule via the API: {e}")
                print(
                    "Assign it in the dashboard instead:\n"
                    "  1. Open https://cloud.livekit.io -> Telephony -> Phone Numbers\n"
                    f"  2. On {PHONE_NUMBER}, open the more menu (three dots) -> Assign dispatch rule\n"
                    f"  3. Select '{RULE_NAME}' ({rule.sip_dispatch_rule_id}) and Save\n"
                    "Then re-run this script; it should report the rule and change nothing."
                )
                sys.exit(1)
            print("assigned:", describe_number(number))
    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="create and assign the rule (default: read-only)")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.apply))
    except (RuntimeError, api.TwirpError) as e:
        sys.exit(f"error: {e}")
