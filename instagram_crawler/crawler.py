# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import argparse
import json
import sys
from io import open
import time
import os
from instagram_crawler.inscrawler import InsCrawler
import urllib.request
from instagram_crawler.inscrawler.settings import override_settings
from instagram_crawler.inscrawler.settings import prepare_override_settings


def usage():
    return """
        python crawler.py posts -u cal_foodie -n 100 -o ./output
        python crawler.py posts_full -u cal_foodie -n 100 -o ./output
        python crawler.py profile -u cal_foodie -o ./output
        python crawler.py profile_script -u cal_foodie -o ./output
        python crawler.py hashtag -t taiwan -o ./output

        The default number for fetching posts via hashtag is 100.
    """


def get_posts_by_user(username, number, detail, debug):
    ins_crawler = InsCrawler(has_screen=debug)
    return ins_crawler.get_user_posts(username, number, detail)


def get_profile(username):
    ins_crawler = InsCrawler()
    return ins_crawler.get_user_profile(username)


def get_profile_from_script(username):
    ins_cralwer = InsCrawler()
    return ins_cralwer.get_user_profile_from_script_shared_data(username)


def get_posts_by_hashtag(tag, number, debug):
    ins_crawler = InsCrawler(has_screen=debug)
    return ins_crawler.get_latest_posts_by_tag(tag, number)


def arg_required(args, fields=[]):
    for field in fields:
        if not getattr(args, field):
            parser.print_help()
            sys.exit()


def output(data, filepath):
    out = json.dumps(data, ensure_ascii=False)
    if filepath:
        with open(filepath, "w", encoding="utf8") as f:
            f.write(out)
    else:
        print(out)
def imagedownload(file_path, hashtag):
    f = open(file_path, "r", encoding="utf-8")
    line = f.readline()
    posts = json.loads(line)
    for i, post in enumerate(posts):
        print(post["img_url"])
        urllib.request.urlretrieve(post["img_url"], "./data/"+hashtag+"/"+str(i) + ".jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instagram Crawler", usage=usage())

    hashtags = ["sushi"]
    for hashtag in hashtags:
        if not os.path.exists("./data/"+hashtag):
            os.makedirs("./data/"+hashtag)
        output(get_posts_by_hashtag(hashtag, 10, None), "./data/"+hashtag+"/output.txt")
        imagedownload("./data/"+hashtag+"/output.txt", hashtag)

