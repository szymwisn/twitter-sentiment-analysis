# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class Tweet(scrapy.Item):
    author_name = scrapy.Field()
    author_id = scrapy.Field()
    sentiment = scrapy.Field()
    date = scrapy.Field()
    emails = scrapy.Field()
    hashtags = scrapy.Field()
    language = scrapy.Field()
    likes = scrapy.Field()
    location = scrapy.Field()
    mentions = scrapy.Field()
    quotes = scrapy.Field()
    retweets = scrapy.Field()
    source = scrapy.Field()
    text = scrapy.Field()
    time = scrapy.Field()
