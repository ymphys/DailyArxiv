import urllib.request as libreq

def main():
    print("Hello from dailyarxiv!")
    with libreq.urlopen('http://export.arxiv.org/api/query?search_query=all:electron+AND+all:beam+AND+all:propagation&start=0&max_results=5') as url:
        r = url.read()
    print(r)


if __name__ == "__main__":
    main()
