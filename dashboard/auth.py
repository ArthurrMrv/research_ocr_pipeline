"""Supabase authentication for the Streamlit dashboard.

Login-only flow — accounts are created manually in the Supabase dashboard.
"""

import os
import time

import streamlit as st
from supabase_auth.errors import AuthApiError
from supabase import Client, create_client


def get_secret(key: str) -> str:
    """Read a secret from st.secrets (Streamlit Cloud) or os.environ (local dev)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        pass
    value = os.environ.get(key)
    if value is None:
        raise RuntimeError(
            f"Required secret '{key}' is not set in st.secrets or environment variables."
        )
    return value


@st.cache_resource
def _get_auth_client() -> Client:
    """Supabase client using the anon key (for auth operations only)."""
    url = get_secret("SUPABASE_URL")
    anon_key = get_secret("SUPABASE_ANON_KEY")
    return create_client(url, anon_key)


def _render_login_form() -> None:
    """Render login form. Sets session_state on success."""
    st.title("Pipeline Dashboard")
    st.markdown("Please log in to continue.")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if not email or not password:
            st.error("Please enter both email and password.")
            return
        try:
            response = _get_auth_client().auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            st.session_state["auth_user"] = response.user
            st.session_state["auth_session"] = response.session
            st.rerun()
        except AuthApiError as exc:
            st.error(f"Login failed: {exc.message}")
        except Exception:
            st.error("An unexpected error occurred. Please try again.")


def require_auth() -> None:
    """Gate function. Call at the top of every page.

    Renders the login form and stops execution if the user is not authenticated.
    Checks token expiry to invalidate stale sessions.
    """
    session = st.session_state.get("auth_session")
    if session is None or session.expires_at <= int(time.time()):
        st.session_state.pop("auth_user", None)
        st.session_state.pop("auth_session", None)
        _render_login_form()
        st.stop()


def logout() -> None:
    """Clear auth state and invalidate the server-side session."""
    try:
        _get_auth_client().auth.sign_out()
    except Exception:
        pass  # best-effort; clear local state regardless
    for key in ("auth_user", "auth_session"):
        st.session_state.pop(key, None)
