# test_bot.py - Simple test version
import asyncio
from jupiter_market_maker import JupiterMarketMaker

async def test_quote():
    """Test basic quote functionality"""
    PRIVATE_KEY = "YOUR_TEST_PRIVATE_KEY"  # Use a test wallet
    
    async with JupiterMarketMaker(PRIVATE_KEY) as bot:
        # Test SOL to USDC quote
        quote = await bot.get_quote(
            bot.mints["SOL"],
            bot.mints["USDC"],
            1000000  # 0.001 SOL
        )
        
        if quote:
            print("Quote received successfully!")
            print(f"Input: {quote.get('inAmount', 0)}")
            print(f"Output: {quote.get('outAmount', 0)}")
            print(f"Price impact: {quote.get('priceImpactPct', 0)}")
        else:
            print("Failed to get quote")

if __name__ == "__main__":
    asyncio.run(test_quote())
