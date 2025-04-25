import streamlit as st
import pandas as pd
import plotly.express as px
import io
import base64
import matplotlib.pyplot as plt
import requests
import json

st.set_page_config(
    page_title="Retrospective Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("Team Retrospective Analysis Tool")
st.markdown("Upload multiple retrospective CSV files to analyze and compare feedback across team retrospectives.")

def compare_retrospectives(file_objects, min_votes, max_votes):
    """
    Process multiple retrospective CSV files and consolidate feedback with vote counts.
    
    Args:
        file_objects: List of uploaded file objects
        min_votes: Minimum vote threshold for filtering
        max_votes: Maximum vote threshold for filtering
        
    Returns:
        List of tuples containing (feedback, task_id, votes)
    """
    feedback_counts = {}
    feedback_tasks = {}  # Dictionary to store associated task numbers
    processing_results = []

    for uploaded_file in file_objects:
        try:
            # Convert to string content
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.split('\n')
            
            # Find the header row
            header_index = next((i for i, line in enumerate(lines) if "Type,Description,Votes" in line), None)
            if header_index is None:
                processing_results.append(f"⚠️ Warning: Skipping {uploaded_file.name} - Required columns not found.")
                continue
                
            # Read CSV content after header
            df = pd.read_csv(io.StringIO(content), skiprows=header_index)
            
            # Check for required columns
            if 'Description' not in df.columns or 'Votes' not in df.columns:
                processing_results.append(f"⚠️ Warning: Skipping {uploaded_file.name} - Required columns missing after header detection.")
                continue
                
            # Process feedback and votes
            df = df[['Description', 'Votes']].dropna()
            df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0).astype(int)
            
            for _, row in df.iterrows():
                feedback = row['Description']
                votes = row['Votes']
                
                if feedback in feedback_counts:
                    feedback_counts[feedback] += votes
                else:
                    feedback_counts[feedback] = votes
            
            # Look for Work Items section
            work_items_header = next((i for i, line in enumerate(lines) 
                                    if "Feedback Description,Work Item Title,Work Item Type,Work Item Id," in line), None)
            
            if work_items_header is not None:
                work_items_df = pd.read_csv(io.StringIO(content), skiprows=work_items_header)
                
                if 'Feedback Description' in work_items_df.columns and 'Work Item Id' in work_items_df.columns:
                    for _, row in work_items_df.iterrows():
                        feedback_desc = row['Feedback Description']
                        work_item_id = row['Work Item Id']
                        if pd.notna(feedback_desc) and pd.notna(work_item_id):
                            feedback_tasks[feedback_desc] = work_item_id
            
            processing_results.append(f"✅ Successfully processed {uploaded_file.name}")
            
        except Exception as e:
            processing_results.append(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    if not feedback_counts:
        return [("No valid feedback found.", None, 0)], processing_results
    
    filtered_feedback = [(feedback, feedback_tasks.get(feedback, None), votes)
                         for feedback, votes in feedback_counts.items()
                         if min_votes <= votes <= max_votes]
    
    # Sort by votes in descending order
    filtered_feedback.sort(key=lambda x: x[2], reverse=True)
    
    return filtered_feedback, processing_results

def create_dataframe_from_results(feedback_results):
    """Convert feedback results to a pandas DataFrame for visualization and export"""
    data = {
        "Feedback": [item[0] for item in feedback_results],
        "Task ID": [str(item[1]) if item[1] else "None" for item in feedback_results],
        "Votes": [item[2] for item in feedback_results]
    }
    return pd.DataFrame(data)

def generate_download_link(df, filename, link_text):
    """Generate a download link for a DataFrame as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{filename}">{link_text}</a>'

# Sidebar for file upload and filtering controls
with st.sidebar:
    st.header("Controls")
    
    uploaded_files = st.file_uploader(
        "Upload Retrospective CSV Files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files containing retrospective data"
    )
    
    st.subheader("Filter Settings")
    min_votes = st.slider("Minimum Votes", 0, 100, 1)
    max_votes = st.slider("Maximum Votes", min_votes, 100, 50)
    
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} file(s)")
    else:
        st.warning("Please upload at least one CSV file")

# Main content area
if not uploaded_files:
    st.info("👈 Please upload retrospective CSV files using the sidebar to begin analysis")
    
    # Show example of expected format
    st.subheader("Expected CSV Format")
    st.markdown(""" 
    Your CSV files should include columns for feedback description and votes, with format like:
    ```
    Type,Description,Votes
    Went Well,The team was collaborative,5
    Needs Improvement,Documentation is lacking,3
    ```
    
    The tool will also recognize associated tasks when formatted as:
    ```
    Feedback Description,Work Item Title,Work Item Type,Work Item Id,
    Documentation is lacking,Improve Docs,Task,12345
    ```
    """)
    
else:
    # Process the uploaded files when the analyze button is clicked
    analyze_button = st.button("Analyze Retrospectives", type="primary")
    
    if analyze_button:
        with st.spinner("Processing retrospective data..."):
            feedback_results, processing_logs = compare_retrospectives(
                uploaded_files, min_votes, max_votes
            )
            
            # Save results to session state for later use in the AI assistant
            st.session_state.retro_feedback = feedback_results
            
            # Show processing results
            with st.expander("Processing Logs", expanded=True):
                for log in processing_logs:
                    st.write(log)
            
            # Convert to DataFrame for easier handling
            results_df = create_dataframe_from_results(feedback_results)
            
            if len(results_df) == 0 or (len(results_df) == 1 and "No valid feedback found" in results_df["Feedback"].iloc[0]):
                st.error("No feedback items found within the selected vote range. Try adjusting your filters.")
            else:
                # Display the results
                st.subheader(f"Consolidated Feedback ({len(results_df)} items)")
                st.dataframe(
                    results_df,
                    column_config={
                        "Feedback": st.column_config.TextColumn("Feedback"),
                        "Task ID": st.column_config.TextColumn("Task ID"),
                        "Votes": st.column_config.NumberColumn("Votes")
                    },
                    use_container_width=True
                )
                
                # Visualization section
                st.subheader("Feedback Visualization")
                
                # Only show top 15 items in chart to avoid overcrowding
                chart_data = results_df.head(15) if len(results_df) > 15 else results_df
                
                # Create a horizontal bar chart with Plotly
                fig = px.bar(
                    chart_data,
                    x="Votes",
                    y="Feedback",
                    orientation='h',
                    title=f"Top Feedback Items by Vote Count (min: {min_votes}, max: {max_votes})",
                    color="Votes",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution of votes
                st.subheader("Vote Distribution")
                vote_distribution = px.histogram(
                    results_df, 
                    x="Votes",
                    nbins=20,
                    title="Distribution of Votes",
                    labels={"Votes": "Vote Count", "count": "Number of Feedback Items"}
                )
                st.plotly_chart(vote_distribution, use_container_width=True)
                
                # Count items with and without associated tasks
                with_tasks = results_df["Task ID"].apply(lambda x: x != "None").sum()
                without_tasks = len(results_df) - with_tasks
                
                # Create pie chart for task association
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                ax3.pie(
                    [with_tasks, without_tasks],
                    labels=["With Task ID", "Without Task ID"],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#4CAF50', '#FF9800']
                )
                ax3.set_title("Feedback Items With Task Association")
                ax3.axis('equal')
                st.pyplot(fig3)
                
                # Export options
                st.subheader("Export Results")
                export_format = st.radio("Select export format:", ["CSV", "Markdown"])
                
                if export_format == "CSV":
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="retrospective_analysis.csv",
                        mime="text/csv"
                    )
                else:  # Markdown
                    # Generate markdown content
                    markdown_content = "# Retrospective Analysis Results\n\n"
                    markdown_content += f"Filter settings: Min Votes = {min_votes}, Max Votes = {max_votes}\n\n"
                    markdown_content += "| Feedback | Task ID | Votes |\n| --- | --- | --- |\n"
                    for _, row in results_df.iterrows():
                        markdown_content += f"| {row['Feedback']} | {row['Task ID']} | {row['Votes']} |\n"
                    
                    st.download_button(
                        label="Download Markdown",
                        data=markdown_content,
                        file_name="retrospective_analysis.md",
                        mime="text/markdown"
                    )

    # AI Assistant Section (Ensure feedback is available before showing this tab)
    import streamlit as st
    import requests
    import json
    
    # Ensure that the session state for messages is initialized
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [{"role": "assistant", "content": "Hi! I'm your retrospective assistant. How can I help?"}]
    
    # Display previous messages in the chat
    with st.expander("AI Assistant"):
        st.header("🤖 AI Retrospective Assistant")
        st.markdown("Ask questions about feedback, trends, and improvements.")
    
        # Display chat messages
        for msg in st.session_state.ai_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
        # Input for API Key
        api_key = st.text_input("🔑 OpenRouter API Key", type="password", key="ai_api_key")
    
        # Ensure feedback exists in session state before allowing AI assistant functionality
        if "retro_feedback" not in st.session_state or st.session_state.retro_feedback is None:
            st.info("Analyze retrospectives first in the previous tab.")
            st.stop()
    
        # Prepare the DataFrame and context for AI
        df = create_dataframe_from_results(st.session_state.retro_feedback)
        context = "You are a helpful assistant summarizing retrospective feedback:\n"
        for _, row in df.iterrows():
            context += f"- {row['Feedback']} ({row['Votes']} votes){' [Task ID: ' + row['Task ID'] + ']' if row['Task ID'] != 'None' else ''}\n"
    
        # Input field for user prompt
        prompt = st.chat_input("Ask me anything about this retrospective...")
    
        if prompt:
            if not api_key:
                st.error("Please enter an API key to proceed.")
            else:
                # Append user message to session state
                st.session_state.ai_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
    
                # Prepare the assistant response
                with st.chat_message("assistant"):
                    msg_placeholder = st.empty()
                    full_response = ""
    
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
    
                    body = {
                        "model": "openai/gpt-3.5-turbo",
                        "messages": [{"role": "system", "content": context}] +
                                    [m for m in st.session_state.ai_messages if m["role"] != "assistant"],
                        "temperature": 0.7,
                        "max_tokens": 1500,
                        "stream": True
                    }
    
                    try:
                        with requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, stream=True) as response:
                            if response.status_code == 200:
                                for chunk in response.iter_lines():
                                    if chunk:
                                        chunk_str = chunk.decode("utf-8")
                                        if chunk_str.startswith("data:"):
                                            data = json.loads(chunk_str[5:])
                                            delta = data["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                full_response += delta["content"]
                                                msg_placeholder.markdown(full_response + "▌")
                            else:
                                full_response = f"Error: {response.status_code} - {response.text}"
                    except Exception as e:
                        full_response = f"Error: {e}"
    
                    # Update the chat message placeholder with the full response once it's complete
                    msg_placeholder.markdown(full_response)
    
                    # Append the assistant's response to session state
                    st.session_state.ai_messages.append({"role": "assistant", "content": full_response})
