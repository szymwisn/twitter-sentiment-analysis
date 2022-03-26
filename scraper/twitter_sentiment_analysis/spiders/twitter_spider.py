import scrapy
import langdetect
import logging
import time

from scrapy.utils.project import get_project_settings
from selenium import webdriver
from scrapy import Selector
from functools import partial

from twitter_sentiment_analysis.items import Tweet
from twitter_sentiment_analysis.hashtags import hashtags_dict

from twitter_sentiment_analysis.constants import SPIDER_NAME, \
    TWITTER_DOMAIN, ARTICLE_XPATH, HREF_XPATH, TWITTER_HASHTAG, \
    TWITTER_URL, SCROLL_DOWN_SCRIPT, LIVE_SUFFIX

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
        unique_tweets_hrefs = self.collect_unique_tweet_hrefs_dict()
        tweet_requests = self.collect_tweet_requests(unique_tweets_hrefs)

        for i, tweet_request in enumerate(tweet_requests):
            logging.info(
                f'[{i + 1}] Tweet to be scraped: ${str(tweet_request)}'
            )
            yield tweet_request

    def collect_unique_tweet_hrefs_dict(self):
        unique_tweet_hrefs_dict = dict()

        for sentiment, hashtags in hashtags_dict.items():
            for hashtag in hashtags:
                url = TWITTER_HASHTAG + hashtag + LIVE_SUFFIX
                logging.info(f'Executing request: {url}')
                self.driver.get(url)

                time.sleep(self.settings['INITIAL_DELAY_IN_SEC'])

                for _ in range(self.settings['SCROLL_DOWN_COUNT']):
                    self.scroll_down()

                    html = self.driver.page_source
                    response = Selector(text=html)

                    tweet_hrefs = self.extract_tweet_hrefs(response)

                    tweets_for_sentiment = unique_tweet_hrefs_dict \
                        .get(sentiment, [])
                    tweets_for_sentiment.extend(tweet_hrefs)

                    unique_tweet_hrefs_dict[sentiment] = tweets_for_sentiment

        self.driver.quit()

        for sentiment, hashtags in hashtags_dict.items():
            unique_tweet_hrefs_dict[sentiment] = list(
                set(unique_tweet_hrefs_dict[sentiment])
            )

        return unique_tweet_hrefs_dict

    def scroll_down(self):
        self.driver.execute_script(SCROLL_DOWN_SCRIPT)
        time.sleep(self.settings['SCROLL_DOWN_INTERVAL_IN_SEC'])

    def extract_tweet_hrefs(self, response):
        combined_xpath = ARTICLE_XPATH + HREF_XPATH
        hrefs = response.xpath(combined_xpath).getall()
        logging.info(f'No. of tweets extracted: {str(len(hrefs))}')
        return hrefs

    def collect_tweet_requests(self, unique_tweet_hrefs_dict):
        for sentiment, tweet_hrefs in unique_tweet_hrefs_dict.items():
            for tweet_href in tweet_hrefs:
                tweet_url = TWITTER_URL + tweet_href
                yield scrapy.Request(
                    tweet_url,
                    callback=partial(self.parse_tweet, sentiment)
                )

    def parse_tweet(self, sentiment, response):
        data = response.xpath(ARTICLE_XPATH + '//text()').getall()

        tweet = Tweet()

        try:
            tweet['author_name'] = data[0]
            tweet['author_id'] = data[1]

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
            tweet['text'] = text
            tweet['language'] = langdetect.detect(text)
            tweet['hashtags'] = find_hashtags(text)
            tweet['mentions'] = find_mentions(text)
            tweet['emails'] = find_emails(text)

            tweet['sentiment'] = sentiment
        except BaseException as e:
            logging.error(f'{e} for data:\n{str(data)}')

        if self.settings['SHOULD_REPLACE_VALUES']:
            tweet['text'] = replace_hashtags(tweet['text'])
            tweet['text'] = replace_mentions(tweet['text'])
            tweet['text'] = replace_emails(tweet['text'])

        yield tweet
