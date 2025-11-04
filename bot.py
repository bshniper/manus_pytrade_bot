import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
import aiohttp
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from spl.token.instructions import get_associated_token_address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JupiterMarketMaker:
    def __init__(self, private_key: str, rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """
        Initialize the market maker bot
        
        Args:
            private_key: Private key for the Solana wallet
            rpc_url: Solana RPC endpoint
        """
        self.jupiter_api_key = "726e5a01-d7a1-4fa2-b48a-0632e90e7c4b"
        self.quote_url = "https://api.jup.ag/ultra/v6/quote"
        self.swap_url = "https://api.jup.ag/ultra/v6/swap"
        
        # Initialize wallet
        self.keypair = Keypair.from_base58_string(private_key)
        self.wallet_address = self.keypair.pubkey()
        
        # Initialize Solana client
        self.solana_client = AsyncClient(rpc_url)
        
        # Token mint addresses
        self.mints = {
            "SOL": "So11111111111111111111111111111111111111112",
            "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "BONK": "DezXbrgEz4Qd6s4R3L83P5GFe2xF89aYQpAatF2VqaHn",
            "WIF": "9wFFo5zr4vLMUG16Z3H9QeWE5cKpM7KDTuTchb7FxPhx"
        }
        
        # Trading parameters
        self.slippage_bps = 50  # 0.5% slippage
        self.min_profit_threshold = 0.001  # 0.1% minimum profit
        
        # Session for HTTP requests
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        await self.solana_client.close()
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for Jupiter API requests"""
        return {
            "Authorization": f"Bearer {self.jupiter_api_key}",
            "Content-Type": "application/json"
        }
    
    async def get_quote(
        self, 
        input_mint: str, 
        output_mint: str, 
        amount: int
    ) -> Optional[Dict]:
        """
        Get quote from Jupiter Ultra API
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in lamports (for SOL) or token decimals
        
        Returns:
            Quote data or None if failed
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "slippageBps": self.slippage_bps
        }
        
        try:
            async with self.session.get(self.quote_url, params=params, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Quote received: {input_mint} -> {output_mint}")
                    return data
                else:
                    logger.error(f"Quote failed: {response.status} - {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None
    
    async def execute_swap(self, quote_response: Dict) -> Optional[str]:
        """
        Execute swap transaction
        
        Args:
            quote_response: Quote response from Jupiter API
            
        Returns:
            Transaction signature or None if failed
        """
        try:
            # Get swap transaction
            swap_data = {
                "quoteResponse": quote_response,
                "userPublicKey": str(self.wallet_address),
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": "auto"
            }
            
            async with self.session.post(
                self.swap_url, 
                json=swap_data, 
                headers=self.get_headers()
            ) as response:
                if response.status == 200:
                    swap_result = await response.json()
                    swap_transaction = swap_result.get('swapTransaction')
                    
                    if not swap_transaction:
                        logger.error("No swap transaction in response")
                        return None
                    
                    # Deserialize transaction
                    transaction_bytes = bytes.fromhex(swap_transaction)
                    transaction = Transaction.deserialize(transaction_bytes)
                    
                    # Sign and send transaction
                    transaction.sign(self.keypair)
                    signature = await self.solana_client.send_transaction(
                        transaction,
                        [self.keypair]
                    )
                    
                    # Confirm transaction
                    await self.confirm_transaction(signature.value)
                    logger.info(f"Swap executed: {signature.value}")
                    return signature.value
                    
                else:
                    logger.error(f"Swap failed: {response.status} - {await response.text()}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error executing swap: {e}")
            return None
    
    async def confirm_transaction(self, signature: str, timeout: int = 60) -> bool:
        """Wait for transaction confirmation"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = await self.solana_client.get_transaction(
                    signature,
                    encoding="json",
                    max_supported_transaction_version=0
                )
                if result.value:
                    if result.value.transaction.meta.err is None:
                        logger.info(f"Transaction confirmed: {signature}")
                        return True
            except Exception as e:
                logger.debug(f"Waiting for confirmation: {e}")
            
            await asyncio.sleep(2)
        
        logger.error(f"Transaction confirmation timeout: {signature}")
        return False
    
    async def get_token_balance(self, mint: str) -> int:
        """Get token balance for specified mint"""
        try:
            if mint == self.mints["SOL"]:
                # Get SOL balance
                balance = await self.solana_client.get_balance(self.wallet_address)
                return balance.value
            else:
                # Get token balance
                token_account = get_associated_token_address(
                    self.wallet_address, 
                    Pubkey.from_string(mint)
                )
                account_info = await self.solana_client.get_token_account_balance(token_account)
                if account_info.value:
                    return int(account_info.value.amount)
                return 0
        except Exception as e:
            logger.error(f"Error getting balance for {mint}: {e}")
            return 0
    
    async def calculate_profitability(self, pair: Tuple[str, str], amount: int) -> Optional[float]:
        """
        Calculate arbitrage profitability for a trading pair
        
        Args:
            pair: Tuple of (input_mint, output_mint)
            amount: Amount to trade
            
        Returns:
            Profit percentage or None if calculation fails
        """
        input_mint, output_mint = pair
        
        # Get forward quote
        forward_quote = await self.get_quote(input_mint, output_mint, amount)
        if not forward_quote:
            return None
        
        # Get reverse quote
        reverse_amount = int(forward_quote['outAmount'])
        reverse_quote = await self.get_quote(output_mint, input_mint, reverse_amount)
        if not reverse_quote:
            return None
        
        # Calculate profit
        final_amount = int(reverse_quote['outAmount'])
        profit = (final_amount - amount) / amount if amount > 0 else 0
        
        return profit
    
    async def market_make_pair(self, pair: Tuple[str, str], base_amount: int) -> bool:
        """
        Execute market making for a specific pair
        
        Args:
            pair: Trading pair (input_mint, output_mint)
            base_amount: Base amount to trade
            
        Returns:
            Success status
        """
        input_mint, output_mint = pair
        
        # Check balances
        input_balance = await self.get_token_balance(input_mint)
        if input_balance < base_amount:
            logger.warning(f"Insufficient balance for {input_mint}. Have {input_balance}, need {base_amount}")
            return False
        
        # Calculate profitability
        profit = await self.calculate_profitability(pair, base_amount)
        if profit is None:
            logger.error("Failed to calculate profitability")
            return False
        
        if profit < self.min_profit_threshold:
            logger.info(f"Profit too low: {profit:.4%}. Skipping trade.")
            return False
        
        logger.info(f"Potential profit: {profit:.4%}. Executing trade...")
        
        # Execute forward trade
        forward_quote = await self.get_quote(input_mint, output_mint, base_amount)
        if not forward_quote:
            return False
        
        forward_sig = await self.execute_swap(forward_quote)
        if not forward_sig:
            return False
        
        # Wait a bit for the transaction to settle
        await asyncio.sleep(5)
        
        # Execute reverse trade
        output_balance = await self.get_token_balance(output_mint)
        if output_balance == 0:
            logger.error("No output tokens received")
            return False
        
        reverse_quote = await self.get_quote(output_mint, input_mint, output_balance)
        if not reverse_quote:
            return False
        
        reverse_sig = await self.execute_swap(reverse_quote)
        if not reverse_sig:
            return False
        
        # Calculate actual profit
        final_balance = await self.get_token_balance(input_mint)
        actual_profit = (final_balance - input_balance) / input_balance if input_balance > 0 else 0
        
        logger.info(f"Market making completed. Actual profit: {actual_profit:.4%}")
        return True
    
    async def run_market_maker(self, trading_pairs: List[Tuple[str, str]], base_amounts: List[int], interval: int = 30):
        """
        Main market making loop
        
        Args:
            trading_pairs: List of trading pairs to monitor
            base_amounts: List of base amounts for each pair
            interval: Time between iterations in seconds
        """
        logger.info("Starting market maker bot...")
        
        while True:
            try:
                # Check wallet balance first
                sol_balance = await self.get_token_balance(self.mints["SOL"])
                logger.info(f"Current SOL balance: {sol_balance}")
                
                if sol_balance < 1000000:  # Minimum 0.001 SOL
                    logger.error("Insufficient SOL balance for transactions")
                    await asyncio.sleep(interval)
                    continue
                
                # Execute market making for each pair
                for i, pair in enumerate(trading_pairs):
                    if i >= len(base_amounts):
                        base_amount = base_amounts[0]  # Use first amount as default
                    else:
                        base_amount = base_amounts[i]
                    
                    pair_name = f"{self.get_token_name(pair[0])}/{self.get_token_name(pair[1])}"
                    logger.info(f"Processing pair: {pair_name}")
                    
                    success = await self.market_make_pair(pair, base_amount)
                    
                    if success:
                        logger.info(f"Successfully traded {pair_name}")
                    else:
                        logger.info(f"Failed to trade {pair_name}")
                
                logger.info(f"Waiting {interval} seconds before next iteration...")
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(interval)
    
    def get_token_name(self, mint: str) -> str:
        """Get token name from mint address"""
        for name, address in self.mints.items():
            if address == mint:
                return name
        return "UNKNOWN"

# Configuration and main execution
async def main():
    # Replace with your actual private key
    PRIVATE_KEY = "YOUR_PRIVATE_KEY_HERE"  # Never commit real private keys!
    
    # Trading pairs and amounts
    trading_pairs = [
        (SOL_MINT, USDC_MINT),  # SOL/USDC
        (USDC_MINT, SOL_MINT),  # USDC/SOL
        (SOL_MINT, BONK_MINT),  # SOL/BONK
        (BONK_MINT, SOL_MINT),  # BONK/SOL
        (SOL_MINT, WIF_MINT),   # SOL/WIF
        (WIF_MINT, SOL_MINT),   # WIF/SOL
    ]
    
    base_amounts = [
        1000000,  # 0.001 SOL for SOL/USDC
        1000000,  # 0.001 SOL equivalent for USDC/SOL
        1000000,  # 0.001 SOL for SOL/BONK
        100000000,  # 100 BONK for BONK/SOL
        1000000,  # 0.001 SOL for SOL/WIF
        1000000,  # 0.001 WIF for WIF/SOL
    ]
    
    async with JupiterMarketMaker(PRIVATE_KEY) as bot:
        await bot.run_market_maker(trading_pairs, base_amounts, interval=60)

if __name__ == "__main__":
    # Constants for easier reference
    SOL_MINT = "So11111111111111111111111111111111111111112"
    USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    BONK_MINT = "DezXbrgEz4Qd6s4R3L83P5GFe2xF89aYQpAatF2VqaHn"
    WIF_MINT = "9wFFo5zr4vLMUG16Z3H9QeWE5cKpM7KDTuTchb7FxPhx"
    
    # Run the bot
    asyncio.run(main())
