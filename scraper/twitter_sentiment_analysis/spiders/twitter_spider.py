import scrapy
import langdetect
import logging

from twitter_sentiment_analysis.items import Tweet
from twitter_sentiment_analysis.constants import SPIDER_NAME, TWITTER_DOMAIN, ARTICLE_XPATH, HREF_XPATH, TWITTER_HASHTAG, TWITTER_URL
from twitter_sentiment_analysis.utils import find_emails, find_hashtags, find_mentions, replace_emails, replace_hashtags, replace_mentions
from twitter_sentiment_analysis.hashtags import hashtags_dict


class TwitterSpider(scrapy.Spider):
    name = SPIDER_NAME
    allowed_domains = [TWITTER_DOMAIN]

    def start_requests(self):
        for _, hashtags in hashtags_dict.items():
            for hashtag in hashtags:
                url = TWITTER_HASHTAG + hashtag
                logging.info('Processing a hashtag with url: ' + url)
                yield scrapy.Request(
                    url,
                    callback=self.parse_for_hashtag,
                    dont_filter=True
                )

    def parse_for_hashtag(self, response):
        tweet_hrefs = self.get_tweet_hrefs(response)
        tweet_requests = self.collect_tweet_requests(tweet_hrefs)

        for tweet_request in tweet_requests:
            yield tweet_request

        self.parse_next_page()

    def get_tweet_hrefs(self, response):
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
        print(data, '\n\n\n')

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

        print (tweet)


    def parse_next_page(self):
        pass
