import math 
class Solution:


    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0
        elif len(prices) > 1: 
            idx1 = 0 
            idx2 = len(prices) - 1 
            minimum, maximum = self.recursiveMaxProfit(prices, idx1, idx2)
            return prices[maximum] - prices[minimum]
        
    def recursiveMaxProfit(self,prices, idx1, idx2) -> int: 
        if idx2 - idx1 == 0: 
            return idx1, idx2 
        else:
            mid_indx = math.floor((idx1 + idx2)/2) # floor the index 
            minidx, maxidx = self.recursiveMaxProfit(prices, idx1, mid_indx)
            minidx2, maxidx2 = self.recursiveMaxProfit(prices, mid_indx +1, idx2)
            
            profit1 = prices[maxidx] - prices[minidx]
            profit2 = prices[maxidx2] - prices[minidx2] 
            profit3 = prices[maxidx2] - prices[minidx] 
            if profit1 > profit2 and profit1 > profit3:
                return minidx, maxidx 
            elif profit2 > profit1 and profit2 > profit3: 
                return minidx2, maxidx2 
            else:
                return minidx, maxidx2 
