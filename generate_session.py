# =================================================================
# generate_session.py
# Interactive Daily Login Helper for Zerodha Kite Connect
# Usage: python generate_session.py
# =================================================================

import sys
import urllib.parse
from auth.session_manager import get_login_url, generate_session_from_request_token


def main():
    print("=" * 65)
    print("  ZERODHA KITE CONNECT - DAILY SESSION GENERATOR")
    print("=" * 65)

    try:
        login_url, api_key, _ = get_login_url()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print("\n1. Open the following URL in your web browser:")
    print(f"\n   {login_url}\n")
    print("2. Log in with your Zerodha User ID, Password & 2FA.")
    print("3. After login, copy the full redirect URL (or just the request_token).")
    print("-" * 65)

    user_input = input("\nPaste request_token or full redirect URL here: ").strip()

    if not user_input:
        print("\n[ERROR] No input provided. Exiting.")
        sys.exit(1)

    # Extract request_token if full URL was pasted
    if "request_token=" in user_input:
        parsed = urllib.parse.urlparse(user_input)
        params = urllib.parse.parse_qs(parsed.query)
        tokens = params.get("request_token", [])
        if not tokens:
            print("\n[ERROR] Could not extract 'request_token' from the pasted URL.")
            sys.exit(1)
        request_token = tokens[0]
    else:
        request_token = user_input

    try:
        print("\nAuthenticating with Zerodha...")
        kite = generate_session_from_request_token(request_token)
        profile = kite.profile()
        print("\n" + "=" * 65)
        print("  LOGIN SUCCESSFUL!")
        print("=" * 65)
        print(f"  User ID     : {profile.get('user_id', 'N/A')}")
        print(f"  User Name   : {profile.get('user_name', 'N/A')}")
        print(f"  Email       : {profile.get('email', 'N/A')}")
        print(f"  Broker      : {profile.get('broker', 'ZERODHA')}")
        print("-" * 65)
        print("  Session cached to config/.session.json (valid for today).")
        print("=" * 65)
    except Exception as e:
        print(f"\n[ERROR] Authentication failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()