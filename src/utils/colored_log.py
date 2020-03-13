from termcolor import colored


class Logger:

    @staticmethod
    def info(text):
        print(colored(text, 'white'))

    @staticmethod
    def warn(text):
        print(colored(text, 'yellow', attrs=['bold']))

    @staticmethod
    def critical(text):
        print(colored(text, 'red', attrs=['bold']))

    @staticmethod
    def error(text):
        print(colored(text, 'red'))

    @staticmethod
    def debug(text):
        print(colored(text, 'green', attrs=['bold']))




