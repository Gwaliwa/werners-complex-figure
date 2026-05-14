"""Export the real selected profile from the Streamlit app session.

Use this inside your app after the profile has been extracted and before/after Werner or PSO.
This saves the actual profile values used by the workflow. It does not digitize a figure.
"""

import pandas as pd
from core.state import get_session

s = get_session()
profile = s.get("profile")

if not profile:
    raise RuntimeError("No profile found in session. Extract/select a profile first.")

x = profile.get("x")
y = profile.get("y_smooth", profile.get("y", profile.get("y_raw")))

if x is None or y is None:
    raise RuntimeError("Profile exists but does not contain x/y arrays.")

out = pd.DataFrame({
    "x": x,
    "magnetic_value": y,
})

out.to_csv("selected_profile_real_from_session.csv", index=False)
print("Saved selected_profile_real_from_session.csv")
