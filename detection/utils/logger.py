import datetime


class Logger:
    def info(self, str):
        print str

    def log(self, tags, message, *args):
        tag = ":".join(tags)
        header = "{}[{}]:".format(datetime.datetime.now(), tag)
        message = str(message)
        message = message.format(*args)
        print header + message

    def error(self, message="", *args):
        self.log(["ERROR"], message, *args)

    def info(self, message="", *args):
        self.log(["INFO"], message, *args)

    def warn(self, message="", *args):
        self.log(["WARNING"], message, *args)

    def debug(self, message, *args):
        self.log(["DEBUG"], message, *args)


logger = Logger()
