import ssl
import aiohttp
import asyncio
import functools
import json
import logging
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse


try:
    import certifi
    import drwcertifi
except ImportError:
    _has_drwcertifi = False
else:
    _has_drwcertifi = True


if _has_drwcertifi:
    try:
        drwcertifi.update()
    except Exception:
        pass


def create_session() -> aiohttp.ClientSession:
    """Create aiohttp ClientSession with DRW certificate ssl context."""
    # SSL is now configured in the client. This function exists
    # for backwards compatibility.
    return aiohttp.ClientSession()


logging.basicConfig(level=logging.INFO)


class NotificationType(str, Enum):
    TEXT = "text"
    ERROR = "error"
    FILL = "fill"
    ACCOUNT = "account"
    ORDER = "order"
    ORDERBOOKS = "orderbooks"
    TRADE = "trade"
    REPORT = "report"


@dataclass
class Fill:
    timestamp: float
    order_id: float
    display_symbol: str
    px: float
    traded_qty: int
    remaining_qty: int


@dataclass
class OpenOrder:
    order_id: float
    display_symbol: str
    px: float
    qty: int


@dataclass
class Order:
    order_id: float
    display_symbol: str
    px: float
    qty: int
    canceled: bool


@dataclass
class OrderBook:
    timestamp: Optional[float] = None
    bids: Dict[float, int] = field(default_factory=dict)
    asks: Dict[float, int] = field(default_factory=dict)
    best_bid_px: Optional[float] = None
    best_bid_qty: Optional[float] = None
    best_ask_px: Optional[float] = None
    best_ask_qty: Optional[float] = None

    def parse(self) -> None:
        self.bids = {
            float(price): quantity
            for price, quantity in self.bids.items()
            if quantity
        }
        self.asks = {
            float(price): quantity
            for price, quantity in self.asks.items()
            if quantity
        }

        if self.bids:
            self.best_bid_px = max(self.bids.keys())
            self.best_bid_qty = self.bids[self.best_bid_px]

        if self.asks:
            self.best_ask_px = min(self.asks.keys())
            self.best_ask_qty = self.asks[self.best_ask_px]


@dataclass
class Trade:
    timestamp: float
    display_symbol: str
    px: float
    qty: int


class APIError(Exception):
    pass


class Client(ABC):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        game_id: int,
        token: str,
        base_url: str = "https://games.drw",
        admin: bool = False,
    ) -> None:
        self.session = session
        self.web_url, self.api_url, self.ws_uri = _game_urls(base_url, game_id)
        self.cash = 0.0
        self.margin = 0.0
        self.positions: Dict[str, int] = {}
        self.notifications: List[str] = []
        self._order_books: Dict[str, OrderBook] = {}
        self._message_queue: \
            asyncio.Queue[Tuple[NotificationType, Any]] = asyncio.Queue()
        self.session.headers["Authorization"] = f"Bearer {token}"
        if admin:
            self.session.headers["X-Trading-Games-Admin"] = "true"
        self._ssl_validation = _ssl_validation_mode(base_url)


    @property
    def order_books(self) -> Dict[str, OrderBook]:
        """The latest order book for each symbol."""
        return self._order_books

    async def start(self) -> None:
        await self.update_positions()
        await self.update_order_books()
        await self.update_notifications()

        tasks = [
            asyncio.create_task(self.socket_reader()),
            asyncio.create_task(self._handle_messages()),
            asyncio.create_task(self.on_start()),
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _make_request(self, method: str, endpoint: str, **kwargs: Any) -> Any:
        """Make an API request."""
        logging.debug("Calling %s %s with %s", method, endpoint, kwargs)
        url = "/".join((self.api_url, endpoint.lstrip("/"))).rstrip("/")
        async with self.session.request(
            method,
            url,
            ssl=self._ssl_validation,
            **kwargs,
        ) as response:
            # Read the body first because `raise_for_status`
            # closes the response.
            try:
                data = await response.json()
            except aiohttp.ContentTypeError:
                data = None

            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                try:
                    detail = data["detail"]
                except Exception:
                    detail = data or "Unknown"
                raise APIError(detail) from e
            else:
                return data

    _get = functools.partialmethod(_make_request, "get")
    _post = functools.partialmethod(_make_request, "post")

    async def register(self) -> None:
        """Register for this game."""
        await self._post("register")

    async def set_theos(self, new_theos: Dict[str, float]) -> None:
        """Update theos."""
        await self._post("set-theos", json=new_theos)

    async def notify(
        self, notification: str, user_id: Optional[int] = None,
    ) -> None:
        """Send a notification."""
        payload = {
            "message": notification,
            "user_id": user_id,
        }
        await self._post("notifications", json=payload)

    async def end_simulation(self, wait: bool = False) -> None:
        """End the game."""
        await self._post("stop")

        if not wait:
            return

        while True:
            # Poll until the job is no longer running.
            game = await self._get("")
            if game["status"] in ("completed", "failed", "canceled"):
                break
            await asyncio.sleep(1)

    async def get_notifications(self) -> List[str]:
        """List notifications."""
        data = await self._get("notifications")
        return [notification["message"] for notification in data]

    async def get_open_orders(self) -> Dict[int, OpenOrder]:
        """Get open orders for the current user."""
        data = await self._get("orders")

        open_orders = {}
        for order_id, order in data.items():
            open_orders[int(order_id)] = _build_open_order(order)

        return open_orders

    async def get_order_books(self) -> Dict[str, OrderBook]:
        """Get the public order books."""
        data = await self._get('orderbooks')
        return _build_order_books(data)

    async def update_positions(self) -> None:
        """Update positions for the current user."""
        try:
            data = await self._get('account')
        except APIError:
            # Account data is unavailable until the
            # user has registered.
            pass
        else:
            self.cash = data['cash']
            self.margin = data.get('margin', 0.0)
            self.positions = data['positions']

    async def update_order_books(self) -> None:
        """Update the complete order book."""
        self._order_books = await self.get_order_books()

    async def update_notifications(self) -> None:
        """Update the notification history."""
        self.notifications = await self.get_notifications()

    async def send_order(
        self, display_symbol: str, px: float, qty: int, order_type: str
    ) -> OpenOrder:
        """Submit an order."""
        payload = {
            "display_symbol": display_symbol,
            "quantity": qty,
            "price": px,
            "order_type": order_type,
        }
        order = await self._post("order/place", json=payload)
        return _build_open_order(order)

    async def cancel_orders(self, order_ids: List[int]) -> None:
        """Cancel an order."""
        payload = {
            "order_ids": order_ids,
        }
        await self._post("order/cancel", json=payload)

    async def purge_display_symbol(self, display_symbol: str) -> None:
        """Purge all orders for one symbol."""
        payload = {
            "display_symbol": display_symbol,
        }
        await self._post("purge", json=payload)

    async def purge_all(self) -> None:
        """Purge all orders."""
        await self._get("purge-all")

    async def on_notification(self, message: str) -> None:
        logging.info("Notification received: %s", message)

    async def on_error(self, error: str) -> None:
        logging.error("Error from Server: %s", error)
        exit(0)

    async def on_fills(self, new_fills: List[Fill]) -> None:
        pass

    async def on_orderbook_updates(
        self, order_books: Dict[str, OrderBook],
    ) -> None:
        pass

    async def on_all_trade(self, trade: Trade) -> None:
        pass

    async def on_order_update(self, order: Order) -> None:
        pass

    async def on_start(self) -> None:
        pass

    async def socket_reader(self) -> None:
        """Receive websocket messages."""
        async with self.session.ws_connect(self.ws_uri) as websocket:
            async for message in websocket:
                data = json.loads(message.data)
                try:
                    notification_type = NotificationType[data["notification_type"]]
                    data = data["data"]
                except Exception:
                    logging.error("Unrecognized notification payload: %s", data)
                else:
                    self._message_queue.put_nowait((notification_type, data))

    async def _handle_messages(self) -> None:
        """Handle websocket messages."""
        while True:
            notification_type, data = await self._message_queue.get()
            await self._handle_message(notification_type, data)
            self._message_queue.task_done()
            await asyncio.sleep(0)

    async def _handle_message(
        self,
        notification_type: NotificationType,
        data: Any,
    ) -> None:
        """Handle a websocket message."""
        if notification_type == NotificationType.TEXT:
            logging.debug(f"Received Notification: {data}")
            self.notifications.append(data["message"])
            await self.on_notification(data["message"])

        elif notification_type == NotificationType.ERROR:
            logging.debug(f"Received Error: {data}")
            await self.on_error(data)

        elif notification_type == NotificationType.ORDER:
            logging.debug(f"Received Order: {data}")
            order = Order(
                order_id=data["id"],
                display_symbol=data["display_symbol"],
                px=data["price"],
                qty=(
                    data["quantity"] if data["side"] == "BID"
                    else -data["quantity"]
                ),
                canceled=data["canceled"],
            )
            await self.on_order_update(order)

        elif notification_type == NotificationType.ACCOUNT:
            self.cash = data["cash"]
            self.margin = data.get('margin', 0.0)

            self.positions.update(data["positions"])
            self.positions = {
                k: self.positions[k]
                for k in self.positions
                if self.positions[k] != 0
            }

        elif notification_type == NotificationType.FILL:
            fills = [Fill(
                data["timestamp"],
                data["order_id"],
                data["display_symbol"],
                data["price"],
                data["traded_quantity"],
                data["remaining_quantity"],
            )]
            logging.debug(f"Received Fills: {fills}")
            await self.on_fills(fills)

        elif notification_type == NotificationType.ORDERBOOKS:
            _apply_order_book_updates(data, self._order_books)
            logging.debug(f"Updated Order Books: {self._order_books}")
            await self.on_orderbook_updates(self._order_books)

        elif notification_type == NotificationType.TRADE:
            trade = Trade(
                data["timestamp"],
                data["display_symbol"],
                data["price"],
                data["quantity"],
            )
            logging.debug(f"Trade: {trade}")
            await self.on_all_trade(trade)

        elif notification_type == NotificationType.REPORT:
            logging.debug(f"Final Report: {data}")


def _build_order_books(data: Dict[str, Dict[str, Any]]) -> Dict[str, OrderBook]:
    """Create order books from an API response."""
    order_books = {
        display_symbol: OrderBook(
            data[display_symbol]["timestamp"],
            data[display_symbol]["bids"],
            data[display_symbol]["asks"],
        )
        for display_symbol in data
    }
    for display_symbol in order_books:
        order_books[display_symbol].parse()

    return order_books


def _apply_order_book_updates(
    data: Dict[str, Dict[str, Any]],
    order_books: Dict[str, OrderBook],
) -> None:
    """Merge incremental updates into the full order book."""
    for display_symbol, updates in data.items():
        try:
            order_book = order_books[display_symbol]
        except KeyError:
            order_book = OrderBook()
            order_books[display_symbol] = order_book

        order_book.bids.update(_cast_prices(updates['bids']))
        order_book.asks.update(_cast_prices(updates['asks']))
        order_book.parse()


def _cast_prices(data: Dict[str, int]) -> Dict[float, int]:
    """Cast prices in an order book to floats."""
    return {
        float(price): quantity for price, quantity in data.items()
    }


def _build_open_order(order: Dict[str, Any]) -> OpenOrder:
    """Build an open order from an API response."""
    quantity = order["quantity"]
    return OpenOrder(
        order_id=order["id"],
        display_symbol=order["display_symbol"],
        px=order["price"],
        qty=quantity if order["side"] == "BID" else -quantity,
    )


def _game_urls(base_url: str, game_id: int) -> Tuple[str, str, str]:
    """Build http and websocket urls for this game."""
    base_url = base_url.rstrip("/")
    parsed = urlparse(base_url)

    if parsed.scheme == "https":
        ws_scheme = "wss"
    else:
        ws_scheme = "ws"

    game_path = f"games/trading-simulator/{game_id}"
    return (
        f"{base_url}/{game_path}",
        f"{base_url}/api/{game_path}",
        f"{ws_scheme}://{parsed.netloc}/ws/{game_path}"
    )


def _ssl_validation_mode(base_url: str) -> Union[bool, ssl.SSLContext]:
    """Determine the SSL validation mode to use."""
    if _has_drwcertifi:
        # If drw-certifi is installed, use the certifi trust store
        # (to which the drw certs have been added).
        return ssl.create_default_context(cafile=certifi.where())

    parsed = urlparse(base_url)
    if parsed.netloc.endswith('.drw'):
        # If drw-certifi is not installed and this is a .drw
        # domain, bypass ssl.
        logging.debug('Bypassing ssl because drw-certifi not installed.')
        return False

    # Otherwise, use the default system trust store.
    return True
