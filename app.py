import streamlit as st
from email_agent import EmailAgent

st.set_page_config(
    page_title="AI Email Agent", 
    page_icon="📧", 
    layout="wide"
)

st.title("AI Email Agent")
st.markdown("### Sophisticated customer email processing with advanced AI")

# Sidebar with system info
with st.sidebar:
    st.markdown("## System Features")
    st.markdown("""
    **Advanced Classification**  
    15 categories, multi-factor scoring
    
    **Confidence Analysis**  
    Multi-dimensional confidence scoring
    
    **Smart Escalation**  
    Intelligent routing with SLA management
    
    **Context-Aware Responses**  
    Dynamic prompt optimization
    
    **Quality Assessment**  
    5-factor response quality scoring
    """)

# Initialize agent
@st.cache_resource
def get_agent():
    return EmailAgent()

agent = get_agent()

# Main content area
col_input, col_results = st.columns([1, 1])

with col_input:
    st.markdown("#### Email Input")
    
    # Email input
    email_content = st.text_area(
        "Customer Email:", 
        height=200,
        placeholder="Type or paste a customer email here...",
        help="Enter the customer's email content to see our advanced AI in action"
    )
    
    # Processing button
    process_col, clear_col = st.columns([3, 1])
    with process_col:
        process_button = st.button("Process Email", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.rerun()

with col_results:
    st.markdown("#### AI Analysis Results")
    
    if process_button and email_content:
        with st.spinner("🔍 Analyzing email with advanced AI..."):
            result = agent.process_email(email_content)
        
        # Enhanced metrics display
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Category", result['category'].replace('_', ' ').title())
        with metric_col2:
            st.metric("Action", result['action'].upper())
        with metric_col3:
            confidence_pct = int(result.get('confidence', 0) * 100)
            st.metric("Confidence", f"{confidence_pct}%")
        with metric_col4:
            processing_time = result.get('processing_time_ms', 0)
            st.metric("Speed", f"{processing_time}ms")
        
        # Action-specific display
        if result['action'] == 'escalate':
            st.error("🚨 **ESCALATION REQUIRED**")
            st.markdown(f"**Reason:** {result.get('reason', 'Complex query')}")
            
            # Show escalation details if available
            if 'priority' in result:
                priority_col, team_col = st.columns(2)
                with priority_col:
                    st.metric("Priority Level", result.get('priority', 'Unknown'))
                with team_col:
                    team = result.get('assigned_team', 'Unknown').replace('_', ' ').title()
                    st.metric("Assigned Team", team)
                    
        elif result['action'] == 'send':
            st.success("✅ **READY TO SEND**")
            st.markdown("### Generated Response:")
            st.info(result.get('response', 'No response generated'))
            
        else:  # review
            st.warning("👁️ **REQUIRES REVIEW**")
            st.markdown("### Generated Response:")
            st.info(result.get('response', 'No response generated'))
        
        # Enhanced details
        st.markdown("---")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.markdown("#### 📋 Summary")
            st.write(result.get('summary', 'Email processed'))
            
            if 'quality_grade' in result:
                st.markdown("#### 🎯 Quality Grade")
                quality = result['quality_grade'].upper()
                if quality in ['EXCELLENT', 'GOOD']:
                    st.success(f"Grade: {quality}")
                elif quality == 'ACCEPTABLE':
                    st.warning(f"Grade: {quality}")
                else:
                    st.error(f"Grade: {quality}")
        
        with detail_col2:
            st.markdown("#### ⚡ Performance")
            if 'tokens_used' in result:
                st.write(f"**Tokens Used:** {result['tokens_used']:,}")
            if 'processing_time_ms' in result:
                st.write(f"**Processing Time:** {result['processing_time_ms']:,}ms")
            
            # Show confidence breakdown if available
            st.markdown("#### 🎯 AI Confidence")
            confidence_score = result.get('confidence', 0)
            
            if confidence_score >= 0.85:
                st.success(f"Very High ({confidence_score:.2f})")
            elif confidence_score >= 0.70:
                st.info(f"Good ({confidence_score:.2f})")
            elif confidence_score >= 0.50:
                st.warning(f"Moderate ({confidence_score:.2f})")
            else:
                st.error(f"Low ({confidence_score:.2f})")
        
        # Advanced debug info
        with st.expander("🔧 Advanced Debug Information"):
            st.json(result)
    
    elif not email_content:
        st.info("👈 Enter an email in the input area to see AI analysis results")

# Bottom section with examples
st.markdown("---")
st.markdown("## 📝 Try These Example Emails")

# Enhanced examples section
examples_col1, examples_col2, examples_col3 = st.columns(3)

with examples_col1:
    st.markdown("#### 📦 Order & Shipping")
    if st.button("📍 Order Status Inquiry", use_container_width=True):
        st.session_state.example_email = "Hi, I placed order #ORD-789123 three days ago and haven't received any tracking information. Can you please check the status? I'm getting worried it might be lost."
    
    if st.button("🚚 Shipping Delay", use_container_width=True):
        st.session_state.example_email = "My package was supposed to arrive yesterday but it still shows 'in transit'. This is urgent as I need it for a presentation tomorrow. What's happening with my delivery?"

with examples_col2:
    st.markdown("#### 💰 Billing & Refunds")
    if st.button("💳 Billing Issue", use_container_width=True):
        st.session_state.example_email = "I was charged twice for my subscription this month - once on the 1st and again on the 15th. This is clearly an error. I need this duplicate charge refunded immediately."
    
    if st.button("↩️ Return Request", use_container_width=True):
        st.session_state.example_email = "The product I received is completely different from what was advertised. The quality is terrible and it doesn't fit. I want a full refund and return label."

with examples_col3:
    st.markdown("#### ⚙️ Technical & Support")
    if st.button("🐛 Technical Problem", use_container_width=True):
        st.session_state.example_email = "Your app keeps crashing every time I try to access my account dashboard. I've tried reinstalling but the problem persists. This is preventing me from managing my subscription."
    
    if st.button("😠 Complaint", use_container_width=True):
        st.session_state.example_email = "This is absolutely unacceptable! I've been trying to get help for weeks and nobody responds. Your customer service is terrible and I'm considering legal action. This is the worst company I've ever dealt with!"

# Handle example selection
if 'example_email' in st.session_state:
    st.text_area("Selected Example:", st.session_state.example_email, height=100, disabled=True)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📝 Use This Example"):
            # This would need JavaScript to populate the main input, so we'll show a message
            st.success("Copy the text above to the main input area!")
    with col2:
        if st.button("❌ Clear Example"):
            del st.session_state.example_email
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
🤖 Powered by Advanced AI • 15 Category Classification • Multi-Factor Confidence Scoring • Smart Escalation Engine
</div>
""", unsafe_allow_html=True)