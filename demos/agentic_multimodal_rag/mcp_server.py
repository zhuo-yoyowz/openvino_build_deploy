from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
import math

mcp = FastMCP("Smart Retail Tools")

# In-memory cart storage
_cart_items = []

# --- Tool 1: Calculate Paint Cost ---
class PaintCostRequest(BaseModel):
    area: float
    price_per_gallon: float
    add_paint_supply_costs: bool = False

@mcp.tool()
def calculate_paint_cost(req: PaintCostRequest) -> float:
    """
    Calculate the total cost of paint needed for a given area in square feet.

    Parameters:
    - area (float): Area to paint in square feet.
    - price_per_gallon (float): Price per gallon in USD.
    - add_paint_supply_costs (bool, optional): Whether to add $50 for supplies.

    Returns:
    - Total cost as a float.

    Example:
    calculate_paint_cost(area=600, price_per_gallon=29.99, add_paint_supply_costs=True)
    """
    gallons_needed = math.ceil((req.area / 400) * 2)
    total_cost = round(gallons_needed * req.price_per_gallon, 2)
    if req.add_paint_supply_costs:
        total_cost += 50
    return total_cost

# --- Tool 2: Calculate Gallons Needed ---
class GallonRequest(BaseModel):
    area: float

@mcp.tool()
def calculate_paint_gallons(req: GallonRequest) -> int:
    """
    Calculate how many gallons of paint are needed to cover a specific area.

    Parameters:
    - area (float): Area in square feet.

    Returns:
    - Number of gallons as an integer (rounded up).

    Example:
    calculate_paint_gallons(area=600)
    """
    return math.ceil((req.area / 400) * 2)

# --- Tool 3: Add to Cart ---
class AddToCartRequest(BaseModel):
    product_name: str
    quantity: int
    price_per_unit: float

@mcp.tool()
def add_to_cart(req: AddToCartRequest) -> dict:
    """
    Add a product to the shopping cart.

    Parameters:
    - product_name (str): Name of the product.
    - quantity (int): Number of units to add.
    - price_per_unit (float): Unit price in USD.

    Returns:
    - Confirmation message and current cart content.

    Example:
    add_to_cart(product_name="Paintbrush", quantity=2, price_per_unit=5.99)
    """
    item = {
        "product_name": req.product_name,
        "quantity": req.quantity,
        "price_per_unit": req.price_per_unit,
        "total_price": round(req.quantity * req.price_per_unit, 2)
    }

    for existing_item in _cart_items:
        if existing_item["product_name"] == req.product_name:
            existing_item["quantity"] += req.quantity
            existing_item["total_price"] = round(existing_item["quantity"] * existing_item["price_per_unit"], 2)
            return {
                "message": f"Updated {req.product_name} quantity to {existing_item['quantity']}",
                "cart": _cart_items
            }

    _cart_items.append(item)
    return {
        "message": f"Added {req.quantity} {req.product_name} to cart",
        "cart": _cart_items
    }

# --- Tool 4: View Cart ---
@mcp.tool()
def view_cart() -> list:
    """
    View the current contents of the shopping cart.

    Returns:
    - List of all items in the cart with their details.

    Example:
    view_cart()
    """
    return _cart_items

# --- Tool 5: Clear Cart ---
@mcp.tool()
def clear_cart() -> dict:
    """
    Clear all items from the shopping cart.

    Returns:
    - Confirmation message.

    Example:
    clear_cart()
    """
    _cart_items.clear()
    return {"message": "Shopping cart has been cleared"}

# Expose the FastAPI app
app = mcp.sse_app()