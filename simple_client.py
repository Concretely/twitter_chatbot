from trainer import twitterbot


tb = twitterbot.twitterbot()
tb.create_responder()


while True:
    i = input(">")
    print("< {}".format(tb.respond_to(tb.preprocess(i))))