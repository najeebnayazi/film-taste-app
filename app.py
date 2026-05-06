
import streamlit as st
import pandas as pd
import plotly.express as px

# ─────────────────────────────────────────
# PAGE CONFIG
# Sets the browser tab title, icon, and layout width
# ─────────────────────────────────────────
st.set_page_config(page_title="My Cinema Portrait", page_icon="🎬", layout="wide")

# ─────────────────────────────────────────
# DATA LOADING & CLEANING
# @st.cache_data means this only runs once — Streamlit remembers
# the result and reuses it instead of reloading every time
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("letterboxd_final.csv")
    
    # Letterboxd exports multi-select fields as comma separated strings
    # e.g. "Drama, Thriller" — this splits them into proper Python lists
    def parse_multi(val):
        if pd.isna(val) or val == "":
            return []
        return [x.strip() for x in str(val).split(",")]
    
    df["Genre"] = df["Genre"].apply(parse_multi)
    df["Director"] = df["Director"].apply(parse_multi)
    df["Country"] = df["Country"].apply(parse_multi)
    
    # Rename Cast to Actor at source so its consistent everywhere downstream
    df["Actor"] = df["Cast"].apply(parse_multi)
    df = df.drop(columns=["Cast"])
    
    # Convert ratings and year to numbers
    # errors="coerce" turns any non-numeric values into NaN instead of crashing
    df["Ratings"] = pd.to_numeric(df["Ratings"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    
    # Parse Watched Date into datetime so we can extract year
    df["Watched Date"] = pd.to_datetime(df["Watched Date"], errors="coerce")
    df["Watched Year"] = df["Watched Date"].dt.year
    
    # Group years into 5-year periods e.g. 2015-2019
    def get_period(year):
        if pd.isna(year):
            return None
        year = int(year)
        start = year - (year % 5)
        return f"{start}-{start+4}"
    
    df["Watched Period"] = df["Watched Year"].apply(get_period)
    
    # Group film release years into decades e.g. 1990s
    df["Decade"] = df["Year"].apply(
        lambda y: f"{int(y)//10*10}s" if pd.notna(y) else None
    )
    
    # Clean Rewatch column — convert to boolean
    # Airtable exports checkboxes as "checked" not "YES"
    df["Rewatch"] = df["Rewatch"].apply(
        lambda x: True if str(x).strip().lower() == "checked" else False
    )
    
    return df

df = load_data()

# ─────────────────────────────────────────
# HELPER FUNCTIONS FOR INSIGHTS
# Each function analyses one section of data and returns
# a list of plain English insight strings
# ─────────────────────────────────────────

def genre_breakdown_insights(genre_counts):
    """Insights about overall genre distribution"""
    insights = []
    top = genre_counts.iloc[0]
    second = genre_counts.iloc[1]
    dominance = int(top["Count"] / second["Count"] * 100 - 100)
    insights.append(f"🎭 **{top['Genre']}** is your most watched genre — {dominance}% more than **{second['Genre']}** in second place.")
    bottom = genre_counts.iloc[-1]
    insights.append(f"📌 **{bottom['Genre']}** is your least explored genre with only {bottom['Count']} films.")
    return insights

def genre_time_insights(genre_period, selected_genres):
    """Insights about genre evolution over time.
    One consolidated insight per genre covering peak and recent trend"""
    insights = []
    for genre in selected_genres:
        g = genre_period[genre_period["Genre"] == genre].sort_values("Watched Period")
        if g.empty or len(g) < 2:
            continue
        peak = g.loc[g["Percentage"].idxmax()]
        recent = g.iloc[-1]["Percentage"]
        previous = g.iloc[-2]["Percentage"]
        if recent > previous:
            trend = "trending up recently"
        elif recent < previous:
            trend = "declining recently"
        else:
            trend = "holding steady recently"
        insights.append(
            f"📈 **{genre}** peaked in **{peak['Watched Period']}** "
            f"at {peak['Percentage']}% of your watching and is {trend}."
        )
    return insights

def decade_insights(decade_counts):
    """Insights about which film eras you gravitate toward"""
    insights = []
    sorted_counts = decade_counts.sort_values("Count", ascending=False)
    top = sorted_counts.iloc[0]
    insights.append(
        f"📅 You gravitate most toward films from the **{top['Decade']}** "
        f"with {top['Count']} films watched."
    )
    least = decade_counts.sort_values("Count", ascending=True).iloc[0]
    insights.append(
        f"📌 The **{least['Decade']}** is your least explored era "
        f"with only {least['Count']} film(s) watched."
    )
    pre80 = decade_counts[
        decade_counts["Decade"].isin(["1920s","1930s","1940s","1950s","1960s","1970s"])
    ]["Count"].sum()
    total = decade_counts["Count"].sum()
    if pre80 / total < 0.05:
        insights.append(f"🎞️ Less than 5% of your watching is pre-1980s — you lean contemporary.")
    return insights

def rating_insights(df):
    """Insights about rating habits and generosity"""
    insights = []
    avg = df["Ratings"].mean()
    five_star = len(df[df["Ratings"] == 5.0])
    one_star = len(df[df["Ratings"] <= 1.5])
    if avg >= 3.8:
        insights.append(
            f"⭐ With an average rating of **{avg:.2f}**, you are a generous rater "
            f"— you mostly watch things you enjoy."
        )
    elif avg <= 3.0:
        insights.append(f"⭐ With an average of **{avg:.2f}**, you are a tough critic.")
    else:
        insights.append(
            f"⭐ Your average rating of **{avg:.2f}** puts you in measured, considered territory."
        )
    insights.append(
        f"🏆 Only **{five_star} films** have earned 5 stars "
        f"— your highest rating is genuinely exclusive."
    )
    if one_star > 0:
        insights.append(
            f"💔 You have given 1.5 stars or below to **{one_star} films** "
            f"— you are not afraid to call something bad."
        )
    return insights

def director_insights(director_counts, director_df):
    """Insights about director preferences and phases"""
    insights = []
    top = director_counts.iloc[0]
    insights.append(
        f"🎬 **{top['Director']}** is your most watched director with {top['Count']} films."
    )
    dir_period = director_df[
        director_df["Director"].isin(director_counts["Director"].tolist())
    ].groupby(["Watched Period", "Director"]).size().reset_index(name="Count")
    if not dir_period.empty:
        peak = dir_period.loc[dir_period["Count"].idxmax()]
        insights.append(
            f"📆 Biggest director phase: **{int(peak['Count'])} {peak['Director']} films** "
            f"watched in {peak['Watched Period']}."
        )
    return insights

def actor_insights(actor_counts, actor_df):
    """Insights about actor preferences and phases"""
    insights = []
    top = actor_counts.iloc[0]
    insights.append(
        f"🌟 **{top['Actor']}** appears most in your watched films "
        f"with {top['Count']} appearances."
    )
    act_period = actor_df[
        actor_df["Actor"].isin(actor_counts["Actor"].tolist())
    ].groupby(["Watched Period", "Actor"]).size().reset_index(name="Count")
    if not act_period.empty:
        peak = act_period.loc[act_period["Count"].idxmax()]
        insights.append(
            f"📆 Biggest actor phase: **{int(peak['Count'])} {peak['Actor']} films** "
            f"watched in {peak['Watched Period']}."
        )
    return insights

def rewatch_insights(rewatch_df, genre_rewatch, director_rewatch):
    """Insights about rewatch behaviour"""
    insights = []
    if rewatch_df.empty:
        return insights
    total_rewatches = len(rewatch_df)
    insights.append(f"🔁 You have rewatched **{total_rewatches} films** in the last 2 years.")
    if not genre_rewatch.empty:
        top_genre = genre_rewatch.iloc[0]
        insights.append(
            f"🎭 **{top_genre['Genre']}** is your most rewatched genre "
            f"with {top_genre['Count']} rewatch entries."
        )
    if not director_rewatch.empty:
        top_dir = director_rewatch.iloc[0]
        insights.append(
            f"🎬 **{top_dir['Director']}** is your most rewatched director "
            f"with {top_dir['Count']} rewatch entries."
        )
    insights.append(
        "⚠️ *Rewatch data is limited to the last 2 years due to a platform migration "
        "from IMDb — these insights reflect recent rewatch behaviour only.*"
    )
    return insights

def display_insights(insights):
    """Render insight strings as styled info callout boxes"""
    for insight in insights:
        st.info(insight)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🎬 My Cinema Portrait")
st.markdown("*A personal deep dive into my watching habits, taste evolution, and film preferences.*")

# Data quality note — surfaces known limitations honestly
st.warning(
    "⚠️ **Data note:** This portrait is based on 1,597 films exported from Letterboxd, "
    "enriched with genre, director and cast data via OMDb (91% coverage). "
    "Rewatch data is available for the last 2 years only due to a prior platform migration from IMDb."
)
st.divider()

# ─────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Films", len(df))
col2.metric("Average Rating", f"{df['Ratings'].mean():.2f} ⭐")
col3.metric("Directors Watched", df["Director"].explode().nunique())
col4.metric("5 Star Films", len(df[df["Ratings"] == 5.0]))

st.divider()

# ─────────────────────────────────────────
# CHART 1 — GENRE BREAKDOWN
# explode() turns each list into individual rows
# so ["Drama", "Thriller"] becomes two separate rows
# ─────────────────────────────────────────
st.subheader("🎭 What I Watch — Genre Breakdown")

genre_df = df.explode("Genre")
genre_df = genre_df[genre_df["Genre"] != ""]
genre_counts = genre_df["Genre"].value_counts().reset_index()
genre_counts.columns = ["Genre", "Count"]

fig1 = px.bar(genre_counts, x="Genre", y="Count",
              color="Count", color_continuous_scale="Viridis")
fig1.update_layout(xaxis_tickangle=45, showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

display_insights(genre_breakdown_insights(genre_counts))
st.divider()

# ─────────────────────────────────────────
# CHART 2 — GENRE OVER TIME
# Normalised by % of total watching per period
# so volume changes dont distort trends
# ─────────────────────────────────────────
st.subheader("📈 How My Taste Evolved — Genre Over Time")

all_genres = genre_df["Genre"].value_counts().index.tolist()
selected_genres = st.multiselect(
    "Select genres to compare",
    options=all_genres,
    default=all_genres[:5]
)

if selected_genres:
    genre_time_df = genre_df[genre_df["Genre"].isin(selected_genres)]
    genre_time_df = genre_time_df.dropna(subset=["Watched Period"])
    genre_period = genre_time_df.groupby(
        ["Watched Period", "Genre"]
    ).size().reset_index(name="Count")
    period_totals = genre_df.dropna(subset=["Watched Period"]).groupby(
        "Watched Period"
    ).size().reset_index(name="Total")
    genre_period = genre_period.merge(period_totals, on="Watched Period")
    genre_period["Percentage"] = (genre_period["Count"] / genre_period["Total"] * 100).round(1)
    genre_period = genre_period.sort_values("Watched Period")
    
    fig2 = px.line(genre_period, x="Watched Period", y="Percentage",
                   color="Genre", markers=True,
                   labels={"Percentage": "% of period watching"})
    fig2.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)
    
    display_insights(genre_time_insights(genre_period, selected_genres))

st.divider()

# ─────────────────────────────────────────
# CHART 3 — ERAS I WATCH — DECADE BREAKDOWN
# ─────────────────────────────────────────
st.subheader("📅 Eras I Watch — Decade Breakdown")

decade_counts = df["Decade"].value_counts().reset_index()
decade_counts.columns = ["Decade", "Count"]
decade_counts = decade_counts.sort_values("Decade")

fig3 = px.bar(decade_counts, x="Decade", y="Count",
              color="Count", color_continuous_scale="Plasma")
st.plotly_chart(fig3, use_container_width=True)

display_insights(decade_insights(decade_counts))
st.divider()

# ─────────────────────────────────────────
# CHART 4 — RATING DISTRIBUTION
# ─────────────────────────────────────────
st.subheader("⭐ How I Rate — Rating Distribution")

rating_counts = df["Ratings"].value_counts().reset_index()
rating_counts.columns = ["Rating", "Count"]
rating_counts = rating_counts.sort_values("Rating")

fig4 = px.bar(rating_counts, x="Rating", y="Count",
              color="Count", color_continuous_scale="Teal")
st.plotly_chart(fig4, use_container_width=True)

display_insights(rating_insights(df))
st.divider()

# ─────────────────────────────────────────
# CHART 5 — TOP DIRECTORS
# horizontal bars work better for long names
# slider lets user control how many to show
# ─────────────────────────────────────────
st.subheader("🎬 Who I Trust — Top Directors")

top_n = st.slider("Show top N directors", 5, 20, 10)
director_df = df.explode("Director")
director_df = director_df[director_df["Director"] != ""]
director_counts = director_df["Director"].value_counts().head(top_n).reset_index()
director_counts.columns = ["Director", "Count"]

fig5 = px.bar(director_counts, x="Count", y="Director",
              orientation="h", color="Count",
              color_continuous_scale="Sunset")
fig5.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig5, use_container_width=True)

display_insights(director_insights(director_counts, director_df))
st.divider()

# ─────────────────────────────────────────
# CHART 6 — DIRECTOR PHASES OVER TIME
# Normalised by % of total watching per period
# ─────────────────────────────────────────
st.subheader("📆 Director Phases — When Did I Watch Them?")

selected_directors = st.multiselect(
    "Select directors to compare",
    options=director_df["Director"].value_counts().head(20).index.tolist(),
    default=director_df["Director"].value_counts().head(5).index.tolist()
)

if selected_directors:
    dir_time_df = director_df[director_df["Director"].isin(selected_directors)]
    dir_time_df = dir_time_df.dropna(subset=["Watched Period"])
    dir_period = dir_time_df.groupby(
        ["Watched Period", "Director"]
    ).size().reset_index(name="Count")
    dir_period_totals = director_df.dropna(subset=["Watched Period"]).groupby(
        "Watched Period"
    ).size().reset_index(name="Total")
    dir_period = dir_period.merge(dir_period_totals, on="Watched Period")
    dir_period["Percentage"] = (dir_period["Count"] / dir_period["Total"] * 100).round(1)
    dir_period = dir_period.sort_values("Watched Period")
    
    fig6 = px.line(dir_period, x="Watched Period", y="Percentage",
                   color="Director", markers=True,
                   labels={"Percentage": "% of period watching"})
    fig6.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# CHART 7 — TOP ACTORS
# ─────────────────────────────────────────
st.subheader("🌟 Who I Watch — Top Actors")

top_n_actors = st.slider("Show top N actors", 5, 20, 10)
actor_df = df.explode("Actor")
actor_df = actor_df[actor_df["Actor"] != ""]
actor_counts = actor_df["Actor"].value_counts().head(top_n_actors).reset_index()
actor_counts.columns = ["Actor", "Count"]

fig7 = px.bar(actor_counts, x="Count", y="Actor",
              orientation="h", color="Count",
              color_continuous_scale="Magma")
fig7.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig7, use_container_width=True)

display_insights(actor_insights(actor_counts, actor_df))
st.divider()

# ─────────────────────────────────────────
# CHART 8 — ACTOR PHASES OVER TIME
# Normalised by % of total watching per period
# ─────────────────────────────────────────
st.subheader("🎭 Actor Phases — When Did I Watch Them?")

selected_actors = st.multiselect(
    "Select actors to compare",
    options=actor_df["Actor"].value_counts().head(20).index.tolist(),
    default=actor_df["Actor"].value_counts().head(5).index.tolist()
)

if selected_actors:
    act_time_df = actor_df[actor_df["Actor"].isin(selected_actors)]
    act_time_df = act_time_df.dropna(subset=["Watched Period"])
    act_period = act_time_df.groupby(
        ["Watched Period", "Actor"]
    ).size().reset_index(name="Count")
    act_period_totals = actor_df.dropna(subset=["Watched Period"]).groupby(
        "Watched Period"
    ).size().reset_index(name="Total")
    act_period = act_period.merge(act_period_totals, on="Watched Period")
    act_period["Percentage"] = (act_period["Count"] / act_period["Total"] * 100).round(1)
    act_period = act_period.sort_values("Watched Period")
    
    fig8 = px.line(act_period, x="Watched Period", y="Percentage",
                   color="Actor", markers=True,
                   labels={"Percentage": "% of period watching"})
    fig8.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig8, use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# CHART 9 — REWATCH INSIGHTS
# Rewatch column is True/False
# We filter to rewatched films only then analyse
# by most rewatched film, genre and director
# Note: only last 2 years of rewatch data available
# ─────────────────────────────────────────
st.subheader("🔁 What I Return To — Rewatch Insights")

rewatch_df = df[df["Rewatch"] == True]

if rewatch_df.empty:
    st.info("No rewatch data found in your export.")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        # Most rewatched films — count appearances in rewatch entries
        st.markdown("**Most Rewatched Films**")
        rewatch_film_counts = rewatch_df["Name"].value_counts().head(10).reset_index()
        rewatch_film_counts.columns = ["Film", "Rewatches"]
        fig9a = px.bar(rewatch_film_counts, x="Rewatches", y="Film",
                       orientation="h", color="Rewatches",
                       color_continuous_scale="Teal")
        fig9a.update_layout(yaxis=dict(autorange="reversed"), height=400)
        st.plotly_chart(fig9a, use_container_width=True)
    
    with col2:
        # Most rewatched genres
        st.markdown("**Most Rewatched Genres**")
        rewatch_genre_df = rewatch_df.explode("Genre")
        rewatch_genre_df = rewatch_genre_df[rewatch_genre_df["Genre"] != ""]
        genre_rewatch = rewatch_genre_df["Genre"].value_counts().head(10).reset_index()
        genre_rewatch.columns = ["Genre", "Count"]
        fig9b = px.bar(genre_rewatch, x="Genre", y="Count",
                       color="Count", color_continuous_scale="Viridis")
        fig9b.update_layout(xaxis_tickangle=45, height=400)
        st.plotly_chart(fig9b, use_container_width=True)
    
    # Most rewatched directors
    st.markdown("**Most Rewatched Directors**")
    rewatch_dir_df = rewatch_df.explode("Director")
    rewatch_dir_df = rewatch_dir_df[rewatch_dir_df["Director"] != ""]
    director_rewatch = rewatch_dir_df["Director"].value_counts().head(10).reset_index()
    director_rewatch.columns = ["Director", "Count"]
    fig9c = px.bar(director_rewatch, x="Count", y="Director",
                   orientation="h", color="Count",
                   color_continuous_scale="Sunset")
    fig9c.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig9c, use_container_width=True)
    
    display_insights(rewatch_insights(rewatch_df, genre_rewatch, director_rewatch))

st.divider()
st.caption("Built with Streamlit · Data from Letterboxd · Enriched via OMDb")
