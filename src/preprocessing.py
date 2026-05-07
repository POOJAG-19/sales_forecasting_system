import pandas as pd

def load_data(path):
    df = pd.read_excel(path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["State", "Date"])

    return df


def fill_missing_dates(df):
    all_states = []

    for state in df["State"].unique():
        temp = df[df["State"] == state].copy()
        temp = temp.set_index("Date").asfreq("D")

        temp["State"] = state
        temp["Total"] = temp["Total"].interpolate()

        all_states.append(temp)

    return pd.concat(all_states).reset_index()