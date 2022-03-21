import scrapy
import langdetect
import logging
import time

from scrapy.utils.project import get_project_settings
from selenium import webdriver
from scrapy import Selector

from twitter_sentiment_analysis.items import Tweet
from twitter_sentiment_analysis.hashtags import hashtags_dict

from twitter_sentiment_analysis.constants import SPIDER_NAME, \
    TWITTER_DOMAIN, ARTICLE_XPATH, HREF_XPATH, TWITTER_HASHTAG, \
    TWITTER_URL, SCROLL_DOWN_SCRIPT

from twitter_sentiment_analysis.utils import find_emails, find_hashtags, \
    find_mentions, replace_emails, replace_hashtags, replace_mentions


class TwitterSpider(scrapy.Spider):
    name = SPIDER_NAME
    allowed_domains = [TWITTER_DOMAIN]

    def __init__(self):
        self.settings = get_project_settings()
        driver_path = self.settings['CHROME_DRIVER_PATH']

        options = webdriver.ChromeOptions()
        for argument in self.settings['WEB_DRIVER_OPTIONS']:
            options.add_argument(argument)

        self.driver = webdriver.Chrome(driver_path, options=options)

    def start_requests(self):
        tweet_hrefs = self.collect_unique_tweet_hrefs()
        logging.info('No. of unique tweets found: ' + str(len(tweet_hrefs)))

        tweet_requests = self.collect_tweet_requests(tweet_hrefs)

        for tweet_request in tweet_requests:
            yield tweet_request

    def collect_unique_tweet_hrefs(self):
        unique_tweet_hrefs = set()

        for category, hashtags in hashtags_dict.items():
            for hashtag in hashtags:
                url = TWITTER_HASHTAG + hashtag
                logging.info('Processing a hashtag with url: ' + url)
                self.driver.get(url)

                time.sleep(self.settings['INITIAL_DELAY_IN_SEC'])

                for _ in range(self.settings['SCROLL_DOWN_COUNT']):
                    self.scroll_down()

                    html = self.driver.page_source
                    response = Selector(text=html)

                    tweet_hrefs = self.extract_tweet_hrefs(response)

                    for tweet_href in tweet_hrefs:
                        unique_tweet_hrefs.add(tweet_href)

        self.driver.quit()

        return unique_tweet_hrefs

    def scroll_down(self):
        self.driver.execute_script(SCROLL_DOWN_SCRIPT)
        time.sleep(self.settings['SCROLL_DOWN_INTERVAL_IN_SEC'])

    def extract_tweet_hrefs(self, response):
        combined_xpath = ARTICLE_XPATH + HREF_XPATH
        hrefs = response.xpath(combined_xpath).getall()
        logging.info('No. of tweets found: ' + str(len(hrefs)))
        return hrefs

    def collect_tweet_requests(self, tweet_hrefs):
        for tweet_href in tweet_hrefs:
            tweet_url = TWITTER_URL + tweet_href
            yield scrapy.Request(tweet_url, callback=self.parse_tweet)

    def parse_tweet(self, response):
        data = response.xpath(ARTICLE_XPATH + '//text()').getall()

        tweet = Tweet()
        tweet['author_name'] = data[0]
        tweet['author_id'] = data[1]

        # TODO: handle "Replying to: #MENTION...", e.g.: https://twitter.com/Hannahmona_S/status/1502304996884647942
        # https://twitter.com/TheDaveWeinbaum/status/1504596642929983494
        dot_separator_index = data.index('·')

        if data[dot_separator_index - 2] == ' from ':
            date_time_index = dot_separator_index - 3
            location_index = dot_separator_index - 1
            tweet['location'] = data[location_index]
        else:
            date_time_index = dot_separator_index - 1

        date_time = data[date_time_index].split('·')
        [time, date] = date_time
        tweet['time'] = time.strip()
        tweet['date'] = date.strip()

        source_index = date_time_index + 2
        tweet['source'] = data[source_index]

        try:
            retweets_index = data.index('Retweets') - 2
            tweet['retweets'] = data[retweets_index]
        except ValueError:
            tweet['retweets'] = 0

        try:
            quotes_index = data.index('Quote Tweets') - 2
            tweet['quotes'] = data[quotes_index]
        except ValueError:
            tweet['quotes'] = 0

        try:
            likes_index = data.index('Likes') - 2
            tweet['likes'] = data[likes_index]
        except ValueError:
            tweet['likes'] = 0

        # TODO: if has an embedded video then date_index - 2 (exclude time, views)
        text = ''.join(data[2:date_time_index])

        tweet['language'] = langdetect.detect(text)

        tweet['hashtags'] = find_hashtags(text)
        tweet['mentions'] = find_mentions(text)
        tweet['emails'] = find_emails(text)

        text = replace_hashtags(text)
        text = replace_mentions(text)
        text = replace_emails(text)

        tweet['text'] = text

        logging.info(tweet)

        yield tweet
