from kis_omega.client import KISClient

def main():
    c = KISClient()
    tok = c.get_token()
    print("OK token (len):", len(tok))

if __name__ == "__main__":
    main()
