"""
Advanced Prompt Engineering System - CORE LOGIC

Sophisticated prompt engineering with category-specific optimization, context-aware selection,
and performance tuning. Goes beyond basic templates to implement dynamic, intelligent
prompt selection based on classification confidence, risk factors, and customer context.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from ..agents.confidence_scorer import ConfidenceAnalysis, RiskFactor
from ..agents.escalation_engine import EscalationDecision

class PromptTemplate(Enum):
    """Different prompt template types for various scenarios"""
    STANDARD = "standard"           # Standard response template
    VIP = "vip"                    # Enhanced template for VIP customers
    HIGH_RISK = "high_risk"        # Careful template for risky situations
    TECHNICAL = "technical"        # Technical detail template
    EMPATHETIC = "empathetic"      # High empathy for negative sentiment
    CONCISE = "concise"           # Brief response template
    DETAILED = "detailed"         # Comprehensive response template

@dataclass
class PromptConfiguration:
    """Configuration for prompt selection and customization"""
    category: str
    template_type: PromptTemplate
    system_prompt: str
    few_shot_examples: List[Dict[str, str]]
    context_variables: Dict[str, str]
    max_tokens: int
    temperature: float
    
class PromptEngineer:
    """
    Advanced prompt engineering system that selects and customizes prompts
    based on email category, confidence analysis, and customer context.
    """
    
    def __init__(self):
        """Initialize prompt engineer with optimized templates"""
        
        # Performance settings for different scenarios
        self.performance_settings = {
            PromptTemplate.STANDARD: {"max_tokens": 300, "temperature": 0.3},
            PromptTemplate.VIP: {"max_tokens": 400, "temperature": 0.2},
            PromptTemplate.HIGH_RISK: {"max_tokens": 250, "temperature": 0.1},
            PromptTemplate.TECHNICAL: {"max_tokens": 500, "temperature": 0.2},
            PromptTemplate.EMPATHETIC: {"max_tokens": 350, "temperature": 0.4},
            PromptTemplate.CONCISE: {"max_tokens": 150, "temperature": 0.3},
            PromptTemplate.DETAILED: {"max_tokens": 600, "temperature": 0.3},
        }
        
        # Initialize category-specific prompts
        self._initialize_category_prompts()
        
        # Initialize few-shot examples
        self._initialize_few_shot_examples()
    
    def select_optimal_prompt(self,
                            category: str,
                            confidence_analysis: ConfidenceAnalysis,
                            escalation_decision: Optional[EscalationDecision],
                            customer_context: Dict,
                            structural_features: Dict) -> PromptConfiguration:
        """
        Select the optimal prompt configuration based on analysis results
        
        Args:
            category: Email classification category
            confidence_analysis: Confidence analysis results
            escalation_decision: Escalation decision (None if not escalating)
            customer_context: Customer context information
            structural_features: Structural features detected
            
        Returns:
            PromptConfiguration with optimized prompt and settings
        """
        
        # Determine template type based on context
        template_type = self._determine_template_type(
            confidence_analysis, escalation_decision, customer_context
        )
        
        # Get base prompt for category and template
        system_prompt = self._get_system_prompt(category, template_type)
        
        # Customize prompt with context
        customized_prompt = self._customize_prompt_with_context(
            system_prompt, customer_context, structural_features, confidence_analysis
        )
        
        # Get relevant few-shot examples
        examples = self._get_few_shot_examples(category, template_type)
        
        # Create context variables for prompt customization
        context_variables = self._create_context_variables(
            customer_context, structural_features, confidence_analysis
        )
        
        # Get performance settings
        settings = self.performance_settings[template_type]
        
        return PromptConfiguration(
            category=category,
            template_type=template_type,
            system_prompt=customized_prompt,
            few_shot_examples=examples,
            context_variables=context_variables,
            max_tokens=settings["max_tokens"],
            temperature=settings["temperature"]
        )
    
    def _determine_template_type(self,
                               confidence_analysis: ConfidenceAnalysis,
                               escalation_decision: Optional[EscalationDecision],
                               customer_context: Dict) -> PromptTemplate:
        """Determine the best template type based on context"""
        
        # VIP customers get special treatment
        if customer_context.get('is_vip', False):
            return PromptTemplate.VIP
        
        # High-risk situations need careful handling
        high_risk_factors = [RiskFactor.LEGAL_TERMS, RiskFactor.COMPLAINT_LANGUAGE, RiskFactor.FINANCIAL_IMPACT]
        if any(factor in confidence_analysis.risk_factors for factor in high_risk_factors):
            return PromptTemplate.HIGH_RISK
        
        # Technical complexity needs detailed responses
        if RiskFactor.TECHNICAL_COMPLEXITY in confidence_analysis.risk_factors:
            return PromptTemplate.TECHNICAL
        
        # High emotion needs empathy
        if RiskFactor.HIGH_EMOTION in confidence_analysis.risk_factors:
            return PromptTemplate.EMPATHETIC
        
        # Low confidence situations need detailed responses
        if confidence_analysis.overall_confidence < 0.7:
            return PromptTemplate.DETAILED
        
        # High confidence can be more concise
        if confidence_analysis.overall_confidence > 0.9:
            return PromptTemplate.CONCISE
        
        # Default to standard
        return PromptTemplate.STANDARD
    
    def _initialize_category_prompts(self):
        """Initialize system prompts for each category and template type"""
        
        self.category_prompts = {
            # CUSTOMER_SUPPORT prompts
            "CUSTOMER_SUPPORT": {
                PromptTemplate.STANDARD: """You are a helpful customer service representative. Respond professionally and empathetically to customer inquiries. Focus on providing clear, actionable solutions while maintaining a friendly tone. Always acknowledge the customer's concern and provide next steps.""",
                
                PromptTemplate.VIP: """You are a premium customer service representative assisting a VIP customer. Provide exceptional, personalized service with extra attention to detail. Offer premium support options when appropriate and ensure the customer feels valued and prioritized.""",
                
                PromptTemplate.HIGH_RISK: """You are a senior customer service representative handling a sensitive inquiry. Be extra careful with your language, avoid making commitments you can't keep, and focus on de-escalation. If unsure about anything, indicate that you'll need to research further.""",
                
                PromptTemplate.EMPATHETIC: """You are a compassionate customer service representative. The customer seems upset or frustrated. Use empathetic language, acknowledge their feelings, and focus on understanding their concern before offering solutions. Show genuine care for their situation.""",
                
                PromptTemplate.TECHNICAL: """You are a technically-knowledgeable customer service representative. The customer has a technical inquiry that requires detailed explanation. Provide clear, step-by-step guidance while avoiding overly complex jargon.""",
                
                PromptTemplate.CONCISE: """You are an efficient customer service representative. Provide a clear, direct response that addresses the customer's inquiry quickly while maintaining professionalism. Be helpful but brief.""",
                
                PromptTemplate.DETAILED: """You are a thorough customer service representative. Provide a comprehensive response that covers all aspects of the customer's inquiry. Include relevant background information, multiple options where applicable, and clear next steps."""
            },
            
            # SALES_ORDER prompts
            "SALES_ORDER": {
                PromptTemplate.STANDARD: """You are a sales order specialist. Help customers with their order-related inquiries efficiently and accurately. Provide order status updates, tracking information, and resolve any order concerns. Be proactive in offering additional assistance.""",
                
                PromptTemplate.VIP: """You are a VIP sales specialist. Provide premium service for this valued customer's order inquiry. Offer expedited processing, priority handling, and additional services that enhance their experience. Ensure they feel like a priority.""",
                
                PromptTemplate.HIGH_RISK: """You are a careful sales order specialist handling a potentially problematic order situation. Verify all information before providing updates, avoid making promises about delivery dates you can't guarantee, and escalate to management if needed."""
            },
            
            # TECHNICAL_ISSUE prompts  
            "TECHNICAL_ISSUE": {
                PromptTemplate.STANDARD: """You are a technical support specialist. Help customers resolve their technical issues with clear, step-by-step guidance. Ask clarifying questions when needed and provide multiple solution options when appropriate.""",
                
                PromptTemplate.TECHNICAL: """You are a senior technical support engineer. The customer has a complex technical issue. Provide detailed technical guidance, including troubleshooting steps, configuration details, and technical explanations. Use appropriate technical terminology while remaining clear.""",
                
                PromptTemplate.VIP: """You are a premium technical support specialist serving a VIP customer. Provide white-glove technical service with immediate escalation paths, direct contact options, and comprehensive technical solutions."""
            },
            
            # BILLING_INQUIRY prompts
            "BILLING_INQUIRY": {
                PromptTemplate.STANDARD: """You are a billing specialist. Help customers understand their billing inquiries with clear explanations of charges, payment options, and account details. Be transparent about billing policies and offer solutions for billing concerns.""",
                
                PromptTemplate.HIGH_RISK: """You are a senior billing specialist handling a sensitive billing dispute. Be extra careful about financial information, verify customer identity, and focus on finding fair resolutions. Avoid making immediate financial commitments without proper authorization.""",
                
                PromptTemplate.VIP: """You are a VIP billing specialist. Provide premium billing support with flexible payment options, expedited processing, and personalized billing solutions for this valued customer."""
            },
            
            # REFUND_REQUEST prompts
            "REFUND_REQUEST": {
                PromptTemplate.STANDARD: """You are a refund specialist. Help customers with their refund requests by explaining the refund process, timelines, and requirements. Be empathetic about their concerns while clearly communicating refund policies.""",
                
                PromptTemplate.HIGH_RISK: """You are a senior refund specialist handling a potentially contentious refund request. Focus on de-escalation, clearly explain refund policies without being defensive, and look for creative solutions within policy guidelines.""",
                
                PromptTemplate.EMPATHETIC: """You are a caring refund specialist. The customer seems disappointed or frustrated with their purchase. Show genuine empathy for their situation, validate their concerns, and work to find the best possible resolution."""
            },
            
            # SHIPPING_INQUIRY prompts
            "SHIPPING_INQUIRY": {
                PromptTemplate.STANDARD: """You are a shipping specialist. Provide accurate shipping information, tracking updates, and delivery estimates. Help customers understand shipping options and resolve any delivery concerns.""",
                
                PromptTemplate.VIP: """You are a VIP shipping specialist. Offer premium shipping options, expedited delivery services, and white-glove shipping support for this valued customer."""
            },
            
            # COMPLAINT_NEGATIVE prompts
            "COMPLAINT_NEGATIVE": {
                PromptTemplate.EMPATHETIC: """You are a specialized complaint resolution specialist. The customer has expressed dissatisfaction or frustration. Your primary goal is to listen, understand, and de-escalate. Show genuine empathy and work toward a satisfactory resolution.""",
                
                PromptTemplate.HIGH_RISK: """You are a senior complaint resolution specialist handling a serious customer complaint. Use careful language, avoid being defensive, acknowledge their concerns, and focus on resolution. Be prepared to escalate if needed."""
            },
            
            # COMPLIMENT_POSITIVE prompts
            "COMPLIMENT_POSITIVE": {
                PromptTemplate.STANDARD: """You are a customer relations specialist receiving positive feedback. Thank the customer warmly for their kind words, share their feedback with the team, and encourage continued engagement.""",
                
                PromptTemplate.CONCISE: """Thank the customer for their positive feedback briefly but warmly. Express appreciation and encourage continued business."""
            },
            
            # Add other categories...
            "PRODUCT_QUESTION": {
                PromptTemplate.STANDARD: """You are a product specialist. Provide detailed, accurate information about products including features, specifications, compatibility, and usage guidance. Help customers make informed decisions."""
            },
            
            "ACCOUNT_ISSUE": {
                PromptTemplate.STANDARD: """You are an account specialist. Help customers with account-related issues including access problems, account settings, and profile updates. Verify identity appropriately before making account changes."""
            },
            
            "LEGAL_COMPLIANCE": {
                PromptTemplate.HIGH_RISK: """You are handling a legal or compliance inquiry. Be extremely careful with your response. Avoid making legal interpretations or commitments. Focus on directing the customer to appropriate legal resources or escalating to the legal team."""
            },
            
            "PARTNERSHIP_BUSINESS": {
                PromptTemplate.DETAILED: """You are a business development specialist. Provide comprehensive information about partnership opportunities, business terms, and next steps for business inquiries. Maintain a professional, business-focused tone."""
            },
            
            "PRESS_MEDIA": {
                PromptTemplate.HIGH_RISK: """You are handling a media inquiry. Be extremely careful with your response. Avoid making statements that could be interpreted as official company positions. Direct media inquiries to the appropriate PR contacts."""
            },
            
            "OTHER_UNCATEGORIZED": {
                PromptTemplate.STANDARD: """You are a general customer service representative. Since this inquiry doesn't fit standard categories, listen carefully to understand the customer's specific need and provide appropriate guidance or routing to the right department."""
            },
            
            "SPAM_PROMOTIONAL": {
                PromptTemplate.CONCISE: """You are handling what appears to be a promotional or spam message. Provide a brief, professional response that acknowledges receipt without encouraging further unsolicited communication."""
            }
        }
    
    def _initialize_few_shot_examples(self):
        """Initialize few-shot examples for each category"""
        
        self.few_shot_examples = {
            "CUSTOMER_SUPPORT": [
                {
                    "input": "Hi, I'm having trouble with my account and can't log in. Can you help?",
                    "output": "I'd be happy to help you with your login issue. Let me guide you through some troubleshooting steps. First, please try resetting your password using the 'Forgot Password' link on the login page. If that doesn't work, please let me know what error message you're seeing, and I'll provide additional assistance."
                },
                {
                    "input": "Your website is confusing and I can't find what I'm looking for. This is frustrating!",
                    "output": "I understand your frustration, and I apologize that our website isn't meeting your expectations. I'm here to help you find exactly what you need. Could you please tell me what specific information or product you're looking for? I'll provide you with direct links and guidance to make this easier."
                }
            ],
            
            "SALES_ORDER": [
                {
                    "input": "Where is my order #12345? I ordered it 5 days ago and haven't received any updates.",
                    "output": "Thank you for contacting us about order #12345. Let me check the current status for you right away. Your order was processed and shipped yesterday via UPS Ground. You should receive tracking information via email shortly. The estimated delivery date is within 2-3 business days. I'll also send you the tracking number directly."
                },
                {
                    "input": "I need to change my shipping address for order #67890 that I just placed.",
                    "output": "I'll be happy to help you update the shipping address for order #67890. Since your order was recently placed, we should be able to make this change before it ships. Please provide me with the correct shipping address, and I'll update it immediately. I'll also send you a confirmation once the change is processed."
                }
            ],
            
            "TECHNICAL_ISSUE": [
                {
                    "input": "The app keeps crashing every time I try to open it. I've tried restarting my phone but it doesn't help.",
                    "output": "I'm sorry to hear about the app crashes. Let's troubleshoot this step by step. First, please try uninstalling and reinstalling the app, which often resolves crash issues. If the problem persists, please let me know your device model and operating system version so I can provide device-specific guidance. I'm here to help until we get this resolved."
                },
                {
                    "input": "I'm getting error code 500 when trying to access my dashboard. What does this mean?",
                    "output": "Error code 500 indicates a server-side issue, which means this isn't something wrong with your device or settings. Our technical team is likely already aware of this issue, but I'll report it immediately to ensure it's being addressed. In the meantime, please try accessing your dashboard again in about 15-20 minutes. I'll follow up with you once we have more information."
                }
            ],
            
            "BILLING_INQUIRY": [
                {
                    "input": "I was charged twice for the same subscription. Can you help me understand why?",
                    "output": "I apologize for the confusion with your billing. Let me investigate this duplicate charge immediately. Based on your account, I can see the charges you're referring to. One appears to be your regular subscription renewal, and the second may be related to [specific reason]. I'll process a refund for the duplicate charge right away and ensure this doesn't happen again."
                },
                {
                    "input": "Can you explain what this $15.99 charge is for on my statement?",
                    "output": "I'd be happy to clarify that charge for you. The $15.99 charge is for [specific service/product] that was activated on [date]. This includes [brief description of what's included]. If you don't recognize this charge or believe it was made in error, please let me know and I'll investigate further and assist with resolution."
                }
            ],
            
            "REFUND_REQUEST": [
                {
                    "input": "I want to return this product. It's not what I expected and doesn't work for my needs.",
                    "output": "I'm sorry to hear the product didn't meet your expectations. I'll be happy to help you with a return. Our return policy allows returns within 30 days of purchase for items in original condition. I'll email you a prepaid return shipping label right away. Once we receive the item, your refund will be processed within 3-5 business days."
                },
                {
                    "input": "I was charged for a service I cancelled last month. I need this refunded immediately.",
                    "output": "I apologize for this billing error after your cancellation. Let me review your account to understand exactly what happened. I can see your cancellation was processed on [date], so this charge should not have occurred. I'm processing a full refund for this charge right now, and it should appear in your account within 3-5 business days. I'll also ensure your cancellation is properly noted to prevent future charges."
                }
            ],
            
            "COMPLAINT_NEGATIVE": [
                {
                    "input": "This is terrible customer service! I've been trying to get help for weeks and no one has responded to my emails!",
                    "output": "I sincerely apologize for the poor experience you've had with our customer service. This is absolutely not the level of service we strive to provide, and I understand your frustration. I'm going to personally ensure your issue is resolved today. Please provide me with your previous case numbers or email details so I can immediately investigate what went wrong and make this right."
                },
                {
                    "input": "Your product is garbage and broke after just one day. I want my money back now!",
                    "output": "I'm truly sorry that our product failed so quickly - that's definitely not the quality we stand behind. I completely understand your frustration and disappointment. Let me make this right immediately. I'm processing a full refund for you right now, and I'll also send you a prepaid return label so you don't incur any additional costs. Your satisfaction is our priority."
                }
            ]
        }
    
    def _get_system_prompt(self, category: str, template_type: PromptTemplate) -> str:
        """Get the system prompt for a specific category and template type"""
        
        category_templates = self.category_prompts.get(category, {})
        
        # Try to get specific template, fall back to standard, then to a default
        if template_type in category_templates:
            return category_templates[template_type]
        elif PromptTemplate.STANDARD in category_templates:
            return category_templates[PromptTemplate.STANDARD]
        else:
            # Default fallback prompt
            return """You are a professional customer service representative. Respond helpfully and courteously to customer inquiries, providing accurate information and appropriate assistance."""
    
    def _customize_prompt_with_context(self,
                                     base_prompt: str,
                                     customer_context: Dict,
                                     structural_features: Dict,
                                     confidence_analysis: ConfidenceAnalysis) -> str:
        """Customize the prompt with contextual information"""
        
        customizations = []
        
        # VIP customer context
        if customer_context.get('is_vip', False):
            vip_indicators = customer_context.get('vip_indicators', [])
            if vip_indicators:
                customizations.append(f"IMPORTANT: This customer is identified as VIP due to: {', '.join(vip_indicators)}. Provide exceptional service.")
        
        # High urgency context
        if customer_context.get('urgency_level') == 'high':
            customizations.append("NOTE: This inquiry appears urgent. Address it with appropriate priority and speed.")
        
        # Multi-channel context
        if customer_context.get('multi_channel', False):
            customizations.append("NOTE: Customer has contacted via multiple channels. Acknowledge this and provide coordinated response.")
        
        # Structural feature context
        helpful_features = []
        if structural_features.get('has_order_number'):
            helpful_features.append("order number provided")
        if structural_features.get('has_account_number'):
            helpful_features.append("account number provided")
        if structural_features.get('has_phone_number'):
            helpful_features.append("phone number available")
        
        if helpful_features:
            customizations.append(f"Available context: {', '.join(helpful_features)}. Use this information effectively.")
        
        # Risk factor context
        if RiskFactor.HIGH_EMOTION in confidence_analysis.risk_factors:
            customizations.append("CAUTION: Customer shows high emotional intensity. Use extra empathy and care.")
        
        if RiskFactor.LEGAL_TERMS in confidence_analysis.risk_factors:
            customizations.append("WARNING: Legal terminology detected. Be extra careful with your response and avoid legal advice.")
        
        if RiskFactor.FINANCIAL_IMPACT in confidence_analysis.risk_factors:
            customizations.append("ALERT: Financial impact detected. Follow financial policies carefully and consider escalation.")
        
        # Combine base prompt with customizations
        if customizations:
            customized_prompt = base_prompt + "\n\n" + "\n".join(customizations)
        else:
            customized_prompt = base_prompt
        
        return customized_prompt
    
    def _get_few_shot_examples(self, category: str, template_type: PromptTemplate) -> List[Dict[str, str]]:
        """Get relevant few-shot examples for the category"""
        
        # Get category-specific examples
        examples = self.few_shot_examples.get(category, [])
        
        # For high-risk templates, filter to more conservative examples
        if template_type == PromptTemplate.HIGH_RISK and len(examples) > 1:
            # Return examples that are more measured and careful
            return examples[:1]  # Just the first, most conservative example
        
        # For VIP templates, we might want to enhance examples (not implemented here for brevity)
        # For concise templates, we might want shorter examples
        
        return examples
    
    def _create_context_variables(self,
                                customer_context: Dict,
                                structural_features: Dict,
                                confidence_analysis: ConfidenceAnalysis) -> Dict[str, str]:
        """Create context variables that can be used in prompt templates"""
        
        variables = {}
        
        # Customer type variable
        if customer_context.get('is_vip', False):
            variables['customer_type'] = 'VIP'
        elif customer_context.get('customer_type') == 'enterprise':
            variables['customer_type'] = 'Enterprise'
        else:
            variables['customer_type'] = 'Standard'
        
        # Urgency variable
        variables['urgency'] = customer_context.get('urgency_level', 'normal')
        
        # Confidence level variable
        if confidence_analysis.overall_confidence > 0.8:
            variables['confidence_level'] = 'high'
        elif confidence_analysis.overall_confidence > 0.6:
            variables['confidence_level'] = 'medium'
        else:
            variables['confidence_level'] = 'low'
        
        # Available context variable
        available_context = []
        if structural_features.get('has_order_number'):
            available_context.append('order_number')
        if structural_features.get('has_account_number'):
            available_context.append('account_number')
        if structural_features.get('has_phone_number'):
            available_context.append('phone_number')
        
        variables['available_context'] = ','.join(available_context) if available_context else 'none'
        
        return variables
    
    def format_prompt_for_llm(self, 
                            prompt_config: PromptConfiguration, 
                            email_content: str,
                            additional_context: Optional[str] = None) -> Dict[str, any]:
        """
        Format the prompt configuration for LLM API call
        
        Returns:
            Dictionary with formatted prompt and parameters for LLM call
        """
        
        # Start with system prompt
        messages = [
            {"role": "system", "content": prompt_config.system_prompt}
        ]
        
        # Add few-shot examples
        for example in prompt_config.few_shot_examples:
            messages.extend([
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]}
            ])
        
        # Add current email with any additional context
        user_message = f"Customer email: {email_content}"
        if additional_context:
            user_message += f"\n\nAdditional context: {additional_context}"
        
        messages.append({"role": "user", "content": user_message})
        
        return {
            "messages": messages,
            "max_tokens": prompt_config.max_tokens,
            "temperature": prompt_config.temperature,
            "metadata": {
                "category": prompt_config.category,
                "template_type": prompt_config.template_type.value,
                "context_variables": prompt_config.context_variables
            }
        }