import re


EMAIL_PATTERN = re.compile(
    r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'
)

MENTION_PATTERN = re.compile(r'(^|[^@\w])@(\w{1,15})\b')

HASHTAG_PATTERN = re.compile(r'(^|[^@\w])#(\w{1,15})\b')


def find_emails(text):
    return re.findall(EMAIL_PATTERN, text)


def find_mentions(text):
    return [mention[1] for mention in re.findall(MENTION_PATTERN, text)]


def find_hashtags(text):
    return [hashtag[1] for hashtag in re.findall(HASHTAG_PATTERN, text)]


def replace_emails(text):
    return re.sub(EMAIL_PATTERN, ' #EMAIL', text)


def replace_mentions(text):
    return re.sub(MENTION_PATTERN, ' #MENTION', text)


def replace_hashtags(text):
    return re.sub(HASHTAG_PATTERN, ' #HASHTAG', text)
