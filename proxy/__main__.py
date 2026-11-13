import uvicorn

from .app import create_app
from .config import LISTEN_HOST, LISTEN_PORT


def main() -> None:
    uvicorn.run(create_app(), host=LISTEN_HOST, port=LISTEN_PORT, log_level="info")


if __name__ == "__main__":
    main()
