from kis_omega.quotes import get_price
import json

def main():
    data = get_price("005930")  # 삼성전자
    print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
