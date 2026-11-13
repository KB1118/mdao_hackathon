import asyncio
import os
import sys

import uvicorn

SIDECAR_HOST = "127.0.0.1"
SIDECAR_PORT = 8787


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    port = int(os.environ.get("HEADROOM_SIDECAR_PORT", SIDECAR_PORT))
    uvicorn.run(
        "headroom.proxy.server:create_app_from_env",
        factory=True,
        host=SIDECAR_HOST,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
