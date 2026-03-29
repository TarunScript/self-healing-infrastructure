# Locust load generator — simulates realistic shopping traffic against Online Boutique
# File: locust/locustfile.py

import random
from locust import HttpUser, task, between

# Product IDs from the Online Boutique catalog
PRODUCT_IDS = [
    "OLJCESPC7Z",  # Sunglasses
    "66VCHSJNUP",  # Tank Top
    "1YMWWN1N4O",  # Watch
    "L9ECAV7KIM",  # Loafers
    "2ZYFJ3GM2N",  # Hairdryer
    "0PUK6V6EV0",  # Candle
    "LS4PSXUNUM",  # Salt & Pepper Shakers
    "9SIQT8TOJO",  # Bamboo Glass Jar
    "6E92ZMYYFZ",  # Mug
]

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD"]


class OnlineBoutiqueUser(HttpUser):
    """Simulates a realistic user browsing and purchasing from Online Boutique."""

    wait_time = between(1, 3)

    @task(10)
    def browse_homepage(self) -> None:
        """Visit the main page — most common action."""
        self.client.get("/", name="GET /")

    @task(5)
    def view_product(self) -> None:
        """View a random product detail page."""
        product_id = random.choice(PRODUCT_IDS)
        self.client.get(f"/product/{product_id}", name="GET /product/[id]")

    @task(3)
    def add_to_cart(self) -> None:
        """Add a random product to cart."""
        product_id = random.choice(PRODUCT_IDS)
        self.client.post(
            "/cart",
            data={"product_id": product_id, "quantity": random.randint(1, 5)},
            name="POST /cart",
        )

    @task(2)
    def view_cart(self) -> None:
        """View the shopping cart."""
        self.client.get("/cart", name="GET /cart")

    @task(1)
    def checkout(self) -> None:
        """Complete a checkout — exercises multiple downstream services."""
        self.client.post(
            "/cart/checkout",
            data={
                "email": "demo@example.com",
                "street_address": "1 Demo St",
                "zip_code": "94043",
                "city": "Mountain View",
                "state": "CA",
                "country": "US",
                "credit_card_number": "4432-8015-6152-0454",
                "credit_card_expiration_month": "1",
                "credit_card_expiration_year": "2030",
                "credit_card_cvv": "672",
            },
            name="POST /cart/checkout",
        )

    @task(2)
    def set_currency(self) -> None:
        """Switch currency — exercises currencyservice."""
        currency = random.choice(CURRENCIES)
        self.client.post(
            "/setCurrency",
            data={"currency_code": currency},
            name="POST /setCurrency",
        )


# Usage:
# Runs automatically via docker-compose.
# Manual: locust -f locustfile.py --host http://localhost:8080 --headless -u 20 -r 5
