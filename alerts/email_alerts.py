import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email_alert(smtp_server, smtp_port, username, password, to_email, subject, body):
    """
    Send an email alert using an SMTP server.

    Parameters:
        smtp_server (str): SMTP server address.
        smtp_port (int): SMTP server port.
        username (str): Sender's email address (also used for login).
        password (str): Password for the sender's email account.
        to_email (str): Recipient's email address.
        subject (str): Subject of the email.
        body (str): Plain text body of the email.
    """
    try:
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Connect to SMTP server and send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()

        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
