"""
Email Service Integration - SERVICE LAYER

Comprehensive email handling including Gmail API integration, SMTP sending,
email parsing, validation, and security features. Supports both reading
incoming emails and sending generated responses.
"""

import asyncio
import aiosmtplib
import email
import re
import base64
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.utils import parseaddr, formataddr
from email import encoders
import html2text

# Gmail API imports (install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client)
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    logging.warning("Gmail API libraries not available. Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class ParsedEmail:
    """Parsed email structure"""
    subject: str
    body: str
    sender_email: str
    sender_name: str
    recipient_email: str
    recipient_name: str
    received_at: datetime
    message_id: str
    in_reply_to: Optional[str] = None
    references: Optional[str] = None
    attachments: List[Dict[str, Any]] = None
    html_body: Optional[str] = None
    is_html: bool = False
    language: str = 'en'
    thread_id: Optional[str] = None

@dataclass 
class EmailResponse:
    """Email response configuration"""
    to_email: str
    to_name: str
    subject: str
    body: str
    from_email: Optional[str] = None
    from_name: Optional[str] = None
    reply_to: Optional[str] = None
    is_html: bool = False
    attachments: List[str] = None
    priority: str = 'normal'  # low, normal, high
    tracking_id: Optional[str] = None

class EmailValidator:
    """Email validation utilities"""
    
    EMAIL_REGEX = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    SUSPICIOUS_PATTERNS = [
        r'click.*here.*immediately',
        r'urgent.*action.*required',
        r'verify.*account.*suspended',
        r'limited.*time.*offer',
        r'act.*now.*or.*lose',
        r'congratulations.*winner',
    ]
    
    SPAM_INDICATORS = [
        'nigeria', 'prince', 'lottery', 'million dollars',
        'inheritance', 'beneficiary', 'transfer funds',
        'urgent assistance', 'confidential business'
    ]
    
    @classmethod
    def is_valid_email(cls, email: str) -> bool:
        """Validate email address format"""
        if not email or len(email) > 254:
            return False
        return bool(cls.EMAIL_REGEX.match(email.strip().lower()))
    
    @classmethod
    def is_suspicious_content(cls, content: str) -> Tuple[bool, List[str]]:
        """Check for suspicious email content"""
        content_lower = content.lower()
        matched_patterns = []
        
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, content_lower):
                matched_patterns.append(pattern)
        
        return len(matched_patterns) > 0, matched_patterns
    
    @classmethod 
    def is_likely_spam(cls, content: str) -> Tuple[bool, float]:
        """Calculate spam likelihood score"""
        content_lower = content.lower()
        spam_score = 0.0
        
        for indicator in cls.SPAM_INDICATORS:
            if indicator in content_lower:
                spam_score += 0.2
        
        # Check for excessive capitalization
        if len([c for c in content if c.isupper()]) / max(len(content), 1) > 0.3:
            spam_score += 0.15
            
        # Check for excessive exclamation marks
        if content.count('!') > 5:
            spam_score += 0.1
            
        # Check for suspicious links
        if re.search(r'http[s]?://[^\s]+', content_lower) and any(word in content_lower for word in ['click', 'urgent', 'now']):
            spam_score += 0.25
            
        return spam_score > 0.5, min(spam_score, 1.0)

class EmailParser:
    """Email parsing and content extraction"""
    
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        
    def parse_email_message(self, raw_message: str) -> ParsedEmail:
        """Parse raw email message into structured format"""
        msg = email.message_from_string(raw_message)
        return self._extract_email_data(msg)
    
    def parse_gmail_message(self, gmail_msg: Dict[str, Any]) -> ParsedEmail:
        """Parse Gmail API message format"""
        headers = {h['name'].lower(): h['value'] for h in gmail_msg.get('payload', {}).get('headers', [])}
        
        subject = headers.get('subject', 'No Subject')
        sender = headers.get('from', '')
        recipient = headers.get('to', '')
        received_at = self._parse_date(headers.get('date'))
        message_id = headers.get('message-id', '')
        in_reply_to = headers.get('in-reply-to')
        references = headers.get('references')
        
        # Extract sender info
        sender_name, sender_email = parseaddr(sender)
        recipient_name, recipient_email = parseaddr(recipient)
        
        # Extract body
        body, html_body, is_html = self._extract_gmail_body(gmail_msg.get('payload', {}))
        
        # Extract attachments
        attachments = self._extract_gmail_attachments(gmail_msg.get('payload', {}))
        
        return ParsedEmail(
            subject=subject,
            body=body,
            sender_email=sender_email,
            sender_name=sender_name,
            recipient_email=recipient_email,
            recipient_name=recipient_name,
            received_at=received_at,
            message_id=message_id,
            in_reply_to=in_reply_to,
            references=references,
            attachments=attachments,
            html_body=html_body,
            is_html=is_html,
            thread_id=gmail_msg.get('threadId')
        )
    
    def _extract_email_data(self, msg: email.message.Message) -> ParsedEmail:
        """Extract data from email.Message object"""
        subject = msg.get('subject', 'No Subject')
        sender = msg.get('from', '')
        recipient = msg.get('to', '')
        received_at = self._parse_date(msg.get('date'))
        message_id = msg.get('message-id', '')
        
        # Parse sender/recipient
        sender_name, sender_email = parseaddr(sender)
        recipient_name, recipient_email = parseaddr(recipient)
        
        # Extract body
        body, html_body, is_html = self._extract_body(msg)
        
        return ParsedEmail(
            subject=subject,
            body=body,
            sender_email=sender_email,
            sender_name=sender_name,
            recipient_email=recipient_email,
            recipient_name=recipient_name,
            received_at=received_at,
            message_id=message_id,
            html_body=html_body,
            is_html=is_html
        )
    
    def _extract_body(self, msg: email.message.Message) -> Tuple[str, Optional[str], bool]:
        """Extract text and HTML body from email message"""
        text_body = None
        html_body = None
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                if 'attachment' not in content_disposition:
                    if content_type == 'text/plain':
                        text_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    elif content_type == 'text/html':
                        html_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                text_body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            elif content_type == 'text/html':
                html_body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        # Convert HTML to text if needed
        if html_body and not text_body:
            text_body = self.html_converter.handle(html_body).strip()
            
        return text_body or '', html_body, bool(html_body and not text_body)
    
    def _extract_gmail_body(self, payload: Dict[str, Any]) -> Tuple[str, Optional[str], bool]:
        """Extract body from Gmail API payload"""
        text_body = ''
        html_body = None
        
        if 'parts' in payload:
            for part in payload['parts']:
                mime_type = part.get('mimeType', '')
                if mime_type == 'text/plain' and 'data' in part.get('body', {}):
                    text_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
                elif mime_type == 'text/html' and 'data' in part.get('body', {}):
                    html_body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
        else:
            mime_type = payload.get('mimeType', '')
            if 'data' in payload.get('body', {}):
                body_data = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
                if mime_type == 'text/plain':
                    text_body = body_data
                elif mime_type == 'text/html':
                    html_body = body_data
        
        # Convert HTML to text if needed
        if html_body and not text_body:
            text_body = self.html_converter.handle(html_body).strip()
            
        return text_body, html_body, bool(html_body and not text_body)
    
    def _extract_gmail_attachments(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract attachment info from Gmail payload"""
        attachments = []
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('filename'):
                    attachment = {
                        'filename': part['filename'],
                        'mime_type': part.get('mimeType'),
                        'size': part.get('body', {}).get('size', 0),
                        'attachment_id': part.get('body', {}).get('attachmentId')
                    }
                    attachments.append(attachment)
        
        return attachments
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse email date string to datetime"""
        if not date_str:
            return datetime.now(timezone.utc)
            
        try:
            # Try parsing with email.utils
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except:
            return datetime.now(timezone.utc)

class GmailService:
    """Gmail API integration service"""
    
    def __init__(self, credentials_path: str = None, token_path: str = None):
        if not GMAIL_AVAILABLE:
            raise ImportError("Gmail API libraries not installed")
            
        self.credentials_path = credentials_path or settings.gmail_credentials_path
        self.token_path = token_path or settings.gmail_token_path
        self.service = None
        self.parser = EmailParser()
    
    def authenticate(self, scopes: List[str] = None) -> bool:
        """Authenticate with Gmail API"""
        scopes = scopes or ['https://www.googleapis.com/auth/gmail.readonly',
                           'https://www.googleapis.com/auth/gmail.send']
        
        creds = None
        
        # Load existing token
        try:
            if self.token_path:
                creds = Credentials.from_authorized_user_file(self.token_path, scopes)
        except Exception as e:
            logger.warning(f"Could not load token: {e}")
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Could not refresh credentials: {e}")
                    return False
            else:
                if not self.credentials_path:
                    logger.error("No credentials file provided for Gmail authentication")
                    return False
                    
                try:
                    flow = Flow.from_client_secrets_file(self.credentials_path, scopes)
                    flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                    
                    auth_url, _ = flow.authorization_url(prompt='consent')
                    logger.info(f"Please visit this URL to authorize: {auth_url}")
                    
                    auth_code = input('Enter the authorization code: ')
                    flow.fetch_token(code=auth_code)
                    creds = flow.credentials
                except Exception as e:
                    logger.error(f"Could not get new credentials: {e}")
                    return False
            
            # Save credentials
            if self.token_path and creds:
                try:
                    with open(self.token_path, 'w') as token:
                        token.write(creds.to_json())
                except Exception as e:
                    logger.warning(f"Could not save token: {e}")
        
        try:
            self.service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail API authentication successful")
            return True
        except Exception as e:
            logger.error(f"Could not build Gmail service: {e}")
            return False
    
    async def get_messages(self, query: str = '', max_results: int = 10) -> List[ParsedEmail]:
        """Get emails from Gmail"""
        if not self.service:
            raise RuntimeError("Gmail service not authenticated")
        
        try:
            # Search for messages
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            parsed_emails = []
            
            for msg in messages:
                try:
                    # Get full message
                    full_msg = self.service.users().messages().get(
                        userId='me', id=msg['id'], format='full'
                    ).execute()
                    
                    # Parse message
                    parsed_email = self.parser.parse_gmail_message(full_msg)
                    parsed_emails.append(parsed_email)
                    
                except Exception as e:
                    logger.error(f"Error parsing message {msg['id']}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(parsed_emails)} emails from Gmail")
            return parsed_emails
            
        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            return []
    
    async def send_email(self, email_response: EmailResponse) -> bool:
        """Send email via Gmail API"""
        if not self.service:
            raise RuntimeError("Gmail service not authenticated")
        
        try:
            # Create message
            message = MIMEMultipart()
            message['to'] = formataddr((email_response.to_name, email_response.to_email))
            message['from'] = formataddr((
                email_response.from_name or settings.default_from_name,
                email_response.from_email or settings.default_from_email
            ))
            message['subject'] = email_response.subject
            
            if email_response.reply_to:
                message['Reply-To'] = email_response.reply_to
            
            # Add priority headers
            if email_response.priority == 'high':
                message['X-Priority'] = '1'
                message['X-MSMail-Priority'] = 'High'
            elif email_response.priority == 'low':
                message['X-Priority'] = '5'
                message['X-MSMail-Priority'] = 'Low'
            
            # Add tracking ID if provided
            if email_response.tracking_id:
                message['X-Tracking-ID'] = email_response.tracking_id
            
            # Add body
            if email_response.is_html:
                message.attach(MIMEText(email_response.body, 'html'))
            else:
                message.attach(MIMEText(email_response.body, 'plain'))
            
            # Add attachments if any
            if email_response.attachments:
                for attachment_path in email_response.attachments:
                    try:
                        with open(attachment_path, 'rb') as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {attachment_path.split("/")[-1]}'
                            )
                            message.attach(part)
                    except Exception as e:
                        logger.warning(f"Could not attach file {attachment_path}: {e}")
            
            # Encode message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send message
            send_result = self.service.users().messages().send(
                userId='me', body={'raw': raw_message}
            ).execute()
            
            logger.info(f"Email sent successfully. Message ID: {send_result['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

class SMTPService:
    """SMTP email sending service"""
    
    def __init__(self, smtp_server: str = None, smtp_port: int = None,
                 username: str = None, password: str = None):
        self.smtp_server = smtp_server or settings.smtp_server
        self.smtp_port = smtp_port or settings.smtp_port
        self.username = username or settings.smtp_username
        self.password = password or settings.smtp_password
        self.use_tls = settings.smtp_use_tls
    
    async def send_email(self, email_response: EmailResponse) -> bool:
        """Send email via SMTP"""
        try:
            # Create message
            message = MIMEMultipart()
            message['To'] = formataddr((email_response.to_name, email_response.to_email))
            message['From'] = formataddr((
                email_response.from_name or settings.default_from_name,
                email_response.from_email or settings.default_from_email
            ))
            message['Subject'] = email_response.subject
            
            if email_response.reply_to:
                message['Reply-To'] = email_response.reply_to
            
            # Add tracking headers
            if email_response.tracking_id:
                message['X-Tracking-ID'] = email_response.tracking_id
            
            # Add body
            if email_response.is_html:
                message.attach(MIMEText(email_response.body, 'html'))
            else:
                message.attach(MIMEText(email_response.body, 'plain'))
            
            # Send email
            if self.use_tls:
                await aiosmtplib.send(
                    message,
                    hostname=self.smtp_server,
                    port=self.smtp_port,
                    username=self.username,
                    password=self.password,
                    start_tls=True
                )
            else:
                await aiosmtplib.send(
                    message,
                    hostname=self.smtp_server,
                    port=self.smtp_port,
                    username=self.username,
                    password=self.password
                )
            
            logger.info(f"Email sent via SMTP to {email_response.to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return False

class EmailService:
    """Main email service orchestrator"""
    
    def __init__(self):
        self.parser = EmailParser()
        self.validator = EmailValidator()
        self.gmail_service = None
        self.smtp_service = None
        
        # Initialize services based on configuration
        if settings.use_gmail_api and GMAIL_AVAILABLE:
            try:
                self.gmail_service = GmailService()
            except Exception as e:
                logger.warning(f"Could not initialize Gmail service: {e}")
        
        if settings.use_smtp:
            self.smtp_service = SMTPService()
    
    async def initialize(self) -> bool:
        """Initialize email services"""
        success = True
        
        if self.gmail_service:
            try:
                if not self.gmail_service.authenticate():
                    logger.warning("Gmail authentication failed")
                    success = False
                else:
                    logger.info("Gmail service initialized successfully")
            except Exception as e:
                logger.error(f"Gmail initialization error: {e}")
                success = False
        
        if self.smtp_service:
            logger.info("SMTP service initialized")
        
        return success
    
    def parse_email(self, raw_email: str) -> ParsedEmail:
        """Parse raw email content"""
        return self.parser.parse_email_message(raw_email)
    
    def validate_email_address(self, email: str) -> bool:
        """Validate email address format"""
        return self.validator.is_valid_email(email)
    
    def analyze_email_content(self, content: str) -> Dict[str, Any]:
        """Analyze email content for security and spam"""
        is_suspicious, suspicious_patterns = self.validator.is_suspicious_content(content)
        is_spam, spam_score = self.validator.is_likely_spam(content)
        
        return {
            'is_suspicious': is_suspicious,
            'suspicious_patterns': suspicious_patterns,
            'is_likely_spam': is_spam,
            'spam_score': spam_score,
            'content_length': len(content),
            'word_count': len(content.split())
        }
    
    async def fetch_emails(self, query: str = 'is:unread', max_results: int = 10) -> List[ParsedEmail]:
        """Fetch emails from configured source"""
        if self.gmail_service:
            return await self.gmail_service.get_messages(query, max_results)
        else:
            logger.warning("No email fetching service configured")
            return []
    
    async def send_response(self, email_response: EmailResponse) -> bool:
        """Send email response using available service"""
        # Validate recipient email
        if not self.validate_email_address(email_response.to_email):
            logger.error(f"Invalid recipient email: {email_response.to_email}")
            return False
        
        # Try Gmail API first, then SMTP
        if self.gmail_service:
            try:
                return await self.gmail_service.send_email(email_response)
            except Exception as e:
                logger.error(f"Gmail sending failed: {e}")
        
        if self.smtp_service:
            try:
                return await self.smtp_service.send_email(email_response)
            except Exception as e:
                logger.error(f"SMTP sending failed: {e}")
        
        logger.error("No email sending service available")
        return False
    
    def format_response_email(self, 
                            original_email: ParsedEmail,
                            response_text: str,
                            subject_prefix: str = "Re: ") -> EmailResponse:
        """Format a response email from original email and generated response"""
        
        # Generate reply subject
        original_subject = original_email.subject
        if not original_subject.lower().startswith('re:'):
            reply_subject = f"{subject_prefix}{original_subject}"
        else:
            reply_subject = original_subject
        
        return EmailResponse(
            to_email=original_email.sender_email,
            to_name=original_email.sender_name,
            subject=reply_subject,
            body=response_text,
            reply_to=original_email.recipient_email,
            from_email=settings.default_from_email,
            from_name=settings.default_from_name,
            tracking_id=f"reply-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
    
    def create_escalation_notification(self,
                                     original_email: ParsedEmail,
                                     escalation_details: Dict[str, Any]) -> EmailResponse:
        """Create escalation notification email"""
        
        subject = f"ESCALATION: {original_email.subject}"
        
        body = f"""
ESCALATED EMAIL NOTIFICATION

Original Email Details:
- From: {original_email.sender_name} <{original_email.sender_email}>
- Subject: {original_email.subject}
- Received: {original_email.received_at}

Escalation Details:
- Reason: {', '.join(escalation_details.get('reasons', []))}
- Priority: {escalation_details.get('priority', 'Unknown')}
- Assigned Team: {escalation_details.get('team', 'Unknown')}
- SLA: {escalation_details.get('sla_hours', 'Unknown')} hours

Original Message:
{original_email.body}

Please handle this email according to escalation procedures.
        """.strip()
        
        return EmailResponse(
            to_email=settings.escalation_email,
            to_name="Support Team",
            subject=subject,
            body=body,
            from_email=settings.default_from_email,
            from_name="AI Email Agent",
            priority='high'
        )


# Global email service instance
email_service = EmailService()