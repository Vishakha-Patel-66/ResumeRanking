import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io

# ----------------------------------------
# Streamlit Config
# ----------------------------------------
st.set_page_config(page_title="AI Resume Screening", layout="wide")
st.title("🤖 AI Resume Screening & Job Matching Dashboard")

st.markdown(
    """
    This app helps HR teams and recruiters to **match resumes with job descriptions** using AI.  
    Upload datasets, explore visualizations, and export ranked results.
    """
)

# ----------------------------------------
# File Upload (Sidebar)
# ----------------------------------------
st.sidebar.header("📂 Upload Datasets")
resumes_file = st.sidebar.file_uploader("Upload Resume Dataset (CSV)", type=["csv"])
jobs_file = st.sidebar.file_uploader("Upload Job Dataset (CSV)", type=["csv"])

if resumes_file and jobs_file:
    resumes = pd.read_csv(resumes_file)
    jobs = pd.read_csv(jobs_file)

    # Clean text function
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
        return text

    resumes["cleaned"] = resumes["Skills"].apply(clean_text)
    jobs["cleaned"] = jobs["Required Skills"].apply(clean_text)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    resume_tfidf = vectorizer.fit_transform(resumes["cleaned"])

    # ----------------------------------------
    # Job Selection
    # ----------------------------------------
    st.subheader("🔍 Job Selection")
    job_title = st.selectbox("Select a Job:", jobs["Job Title"])
    job_index = jobs[jobs["Job Title"] == job_title].index[0]
    job_tfidf = vectorizer.transform([jobs.iloc[job_index]["cleaned"]])

    # Cosine similarity
    cosine_sim = cosine_similarity(resume_tfidf, job_tfidf).flatten()

    # ----------------------------------------
    # Candidate Search
    # ----------------------------------------
    st.subheader("👤 Candidate Search")
    candidate_name = st.text_input("Enter Candidate Name or ID:")

    if candidate_name:
        candidate_match = resumes[
            resumes["Name"].str.contains(candidate_name, case=False) |
            resumes["Resume_ID"].astype(str).str.contains(candidate_name)
        ]
        if not candidate_match.empty:
            idx = candidate_match.index[0]
            score = cosine_sim[idx]
            st.success(f"Candidate **{candidate_match.iloc[0]['Name']}** → Match Score: {round(score*100, 2)}%")
        else:
            st.error("No candidate found!")

    # ----------------------------------------
    # Ranking
    # ----------------------------------------
    ranking = pd.DataFrame({
        "Name": resumes["Name"],
        "Resume_ID": resumes["Resume_ID"],
        "Skills": resumes["Skills"],
        "Score": cosine_sim.round(3)
    })
    ranking = ranking.sort_values(by="Score", ascending=False)

    # ----------------------------------------
    # Top N input for table and histogram (max 50)
    # ----------------------------------------
    st.subheader("⚡ Display Top Resumes")
    max_display = 50  # Fixed maximum
    default_value = 10

    top_n = st.number_input(
        "Select number of Top Resumes to display:",
        min_value=5,
        max_value=max_display,
        value=default_value,
        step=1
    )

    top_ranking = ranking.head(top_n)

    # Job Info
    st.subheader("📌 Job Information")
    st.write("**Job Title:**", jobs.iloc[job_index]["Job Title"])
    st.write("**Required Skills:**", jobs.iloc[job_index]["Required Skills"])

    # Top Matching Resumes Table
    st.subheader("🏆 Top Matching Resumes")
    st.dataframe(top_ranking)

    # Download button
    csv_buffer = io.StringIO()
    top_ranking.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Top Matches as CSV",
        data=csv_buffer.getvalue(),
        file_name="top_matches.csv",
        mime="text/csv",
    )

    # ----------------------------------------
    # Tabs for Visualizations
    # ----------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Histogram", "📈 Line Chart", "⚪ Bubble Chart", "📊 Clustered Bar Chart"]
    )

    # ----------------------------------------
    # 1. Interactive Histogram (Top N)
    # ----------------------------------------
    with tab1:
        st.markdown("### Interactive Histogram of Top Resume Scores")
        st.write("Hover over the bars to see Resume Name and ID.")

        hist_df = top_ranking.copy()
        hist_df["Score_Percent"] = hist_df["Score"] * 100
        hist_df["Hover_Text"] = hist_df.apply(lambda row: f"{row['Name']} (ID:{row['Resume_ID']})", axis=1)

        fig_hist = px.bar(
            hist_df,
            x="Hover_Text",
            y="Score_Percent",
            text="Score_Percent",
            labels={"Score_Percent": "Match Percentage (%)", "Hover_Text": "Resume"},
            title=f"Top {top_n} Resume Rankings for Job: {jobs.iloc[job_index]['Job Title']}",
            color="Score_Percent",
            color_continuous_scale="Viridis"
        )
        fig_hist.update_traces(
            hovertemplate='%{x}<br>Score: %{y:.2f}%',
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )
        fig_hist.update_layout(
            xaxis_tickangle=-45,
            yaxis=dict(range=[0, max(hist_df["Score_Percent"]) + 10]),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ----------------------------------------
    # 2. Line Chart
    # ----------------------------------------
    with tab2:
        line_df = pd.DataFrame({
            "Resume Index": resumes.index,
            "Name": resumes["Name"],
            "ResumeID": resumes["Resume_ID"],
            "Score": cosine_sim
        })
        fig_line = px.line(
            line_df,
            x="Resume Index",
            y="Score",
            markers=True,
            hover_data=["Name", "ResumeID"],
            title=f"Prediction Scores for All Resumes vs Job: {jobs.iloc[job_index]['Job Title']}",
            color_discrete_sequence=["blue"]
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ----------------------------------------
    # 3. Bubble Chart
    # ----------------------------------------
    with tab3:
        num_resumes = st.slider("Select number of resumes:", 5, len(resumes), 20, 5)
        num_jobs = st.slider("Select number of jobs:", 5, len(jobs), 10, 5)

        job_tfidf_all = vectorizer.transform(jobs["cleaned"])
        heatmap_scores = cosine_similarity(resume_tfidf, job_tfidf_all)

        bubble_data = []
        for i in range(num_resumes):
            for j in range(num_jobs):
                bubble_data.append({
                    "Resume": f"Resume_{resumes.iloc[i]['Resume_ID']}",
                    "Job": f"{jobs.iloc[j]['Job Title']}",
                    "Score": heatmap_scores[i, j]
                })

        bubble_df = pd.DataFrame(bubble_data)
        fig_bubble = px.scatter(
            bubble_df,
            x="Job",
            y="Resume",
            size="Score",
            color="Score",
            hover_data=["Score"],
            title="Resume vs Job Match (Bubble Chart)",
            size_max=30,
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # ----------------------------------------
    # 4. Clustered Bar Chart
    # ----------------------------------------
    with tab4:
        num_resumes_bar = st.slider("Resumes for Bar Chart:", 5, len(resumes), 10, 5)
        num_jobs_bar = st.slider("Jobs for Bar Chart:", 5, len(jobs), 5, 5)

        top_resumes = resumes.head(num_resumes_bar)
        top_jobs = jobs.head(num_jobs_bar)

        bar_data = []
        for i in range(len(top_resumes)):
            for j in range(len(top_jobs)):
                bar_data.append({
                    "Resume": f"{top_resumes.iloc[i]['Name']} (ID:{top_resumes.iloc[i]['Resume_ID']})",
                    "Job": top_jobs.iloc[j]["Job Title"],
                    "Score": cosine_similarity(
                        resume_tfidf[i],
                        vectorizer.transform([top_jobs.iloc[j]["cleaned"]])
                    )[0][0]
                })

        bar_df = pd.DataFrame(bar_data)
        fig_bar = px.bar(
            bar_df,
            x="Resume",
            y="Score",
            color="Job",
            barmode="group",
            title="Resume vs Multiple Jobs - Match Scores"
        )
        fig_bar.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.warning("⚠️ Please upload both Resume and Job datasets to continue.")
