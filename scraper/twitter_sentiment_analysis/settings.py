BOT_NAME = 'twitter_sentiment_analysis'

SPIDER_MODULES = ['twitter_sentiment_analysis.spiders']
NEWSPIDER_MODULE = 'twitter_sentiment_analysis.spiders'

CHROME_DRIVER_PATH = 'twitter_sentiment_analysis/chromedriver'
WEB_DRIVER_OPTIONS = ['--enable-javascript']

USER_AGENT = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html'

ROBOTSTXT_OBEY = True

DOWNLOAD_DELAY = 1

# CUSTOM SETTINGS
INITIAL_DELAY_IN_SEC = 8
SCROLL_DOWN_INTERVAL_IN_SEC = 3
SCROLL_DOWN_COUNT = 40

SHOULD_REPLACE_VALUES = False
