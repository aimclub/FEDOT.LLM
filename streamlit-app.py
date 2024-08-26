from web.frontend import fragments as fg


def main():
    fg.init_session()
    fg.side_bar()
    fg.chat()


if __name__ == "__main__":
    main()
