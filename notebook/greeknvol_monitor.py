import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')


class OptionPricer:
    """Black-Scholes Option Pricer with Greeks calculation"""

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        Parameters:
        -----------
        S : float - Spot price
        K : float - Strike price
        T : float - Time to maturity (years)
        r : float - Risk-free rate
        sigma : float - Volatility
        option_type : str - 'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

    def _d1_d2(self):
        """Calculate d1 and d2 from Black-Scholes formula"""
        if self.T <= 0:
            return 0, 0

        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self):
        """Calculate Black-Scholes option price"""
        if self.T <= 0:
            if self.option_type == 'call':
                return max(0, self.S - self.K)
            else:
                return max(0, self.K - self.S)

        d1, d2 = self._d1_d2()

        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

        return price

    def delta(self):
        """Calculate Delta: dV/dS"""
        if self.T <= 0:
            if self.option_type == 'call':
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0

        d1, _ = self._d1_d2()

        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma(self):
        """Calculate Gamma: d²V/dS²"""
        if self.T <= 0:
            return 0

        d1, _ = self._d1_d2()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """Calculate Vega: dV/dsigma (per 1% change)"""
        if self.T <= 0:
            return 0

        d1, _ = self._d1_d2()
        return self.S * norm.pdf(d1) * np.sqrt(self.T) / 100

    def theta(self):
        """Calculate Theta: -dV/dT (per day)"""
        if self.T <= 0:
            return 0

        d1, d2 = self._d1_d2()

        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))

        if self.option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            return (term1 - term2) / 365
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            return (term1 + term2) / 365

    def rho(self):
        """Calculate Rho: dV/dr (per 1% change)"""
        if self.T <= 0:
            return 0

        _, d2 = self._d1_d2()

        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100

    def get_all_greeks(self):
        """Return dictionary with price and all Greeks"""
        return {
            'Price': self.price(),
            'Delta': self.delta(),
            'Gamma': self.gamma(),
            'Vega': self.vega(),
            'Theta': self.theta(),
            'Rho': self.rho()
        }

    @staticmethod
    def implied_volatility(S, K, T, r, market_price, option_type='call',
                          max_iter=100, tolerance=1e-6):
        """
        Calculate implied volatility using Newton-Raphson method

        Returns: implied volatility or None if not converged
        """
        if T <= 0:
            return None

        sigma = 0.3  # Initial guess

        for i in range(max_iter):
            pricer = OptionPricer(S, K, T, r, sigma, option_type)
            price = pricer.price()
            vega = pricer.vega() * 100  # Convert to full vega

            if abs(vega) < 1e-10:
                break

            diff = market_price - price

            if abs(diff) < tolerance:
                return sigma

            sigma += diff / vega

            # Boundary constraints
            sigma = max(0.001, min(sigma, 5.0))

        return sigma


class GreekAnalyzer:
    """Analyze and visualize option Greeks and volatility surfaces"""

    def __init__(self, S=100, K=100, T=0.25, r=0.05, sigma=0.25, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.pricer = OptionPricer(S, K, T, r, sigma, option_type)

    def print_greeks(self):
        """Print all Greeks in formatted table"""
        greeks = self.pricer.get_all_greeks()

        print("\n" + "="*60)
        print(f"BLACK-SCHOLES GREEKS - {self.option_type.upper()}")
        print("="*60)
        print(f"Spot (S):      ${self.S:.2f}")
        print(f"Strike (K):    ${self.K:.2f}")
        print(f"Time (T):      {self.T:.4f} years ({self.T*365:.0f} days)")
        print(f"Rate (r):      {self.r*100:.2f}%")
        print(f"Volatility:    {self.sigma*100:.2f}%")
        print("-"*60)

        for greek, value in greeks.items():
            if greek == 'Price':
                print(f"{greek:12s}: ${value:10.4f}")
            else:
                print(f"{greek:12s}: {value:11.6f}")

        print("="*60 + "\n")

    def plot_volatility_surface(self, strike_range=0.3, maturity_max=1.0):
        """
        Generate 3D volatility surface

        Parameters:
        -----------
        strike_range : float - Range around ATM (e.g., 0.3 = 70% to 130%)
        maturity_max : float - Maximum maturity in years
        """
        # Generate grid
        strikes = np.linspace(self.S * (1 - strike_range),
                             self.S * (1 + strike_range), 20)
        maturities = np.linspace(0.05, maturity_max, 20)

        K_grid, T_grid = np.meshgrid(strikes, maturities)
        IV_grid = np.zeros_like(K_grid)

        # Calculate implied volatility surface (simplified model)
        for i in range(len(maturities)):
            for j in range(len(strikes)):
                # Simplified IV smile: higher for OTM, decreases with time
                moneyness = strikes[j] / self.S
                IV_grid[i, j] = self.sigma + 0.1 * np.abs(moneyness - 1) * np.exp(-maturities[i])

        # Plot
        fig = plt.figure(figsize=(14, 6))

        # 3D Surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(T_grid, K_grid, IV_grid * 100,
                               cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.set_xlabel('Time to Maturity (Years)', fontsize=10)
        ax1.set_ylabel('Strike Price ($)', fontsize=10)
        ax1.set_zlabel('Implied Volatility (%)', fontsize=10)
        ax1.set_title('Implied Volatility Surface', fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

        # Contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(T_grid, K_grid, IV_grid * 100, levels=15, cmap='viridis')
        ax2.set_xlabel('Time to Maturity (Years)', fontsize=10)
        ax2.set_ylabel('Strike Price ($)', fontsize=10)
        ax2.set_title('Volatility Surface - Contour View', fontsize=12, fontweight='bold')
        fig.colorbar(contour, ax=ax2, label='Implied Volatility (%)')

        plt.tight_layout()
        plt.show()

    def plot_greek_sensitivity(self, greek='delta', variable='spot', range_pct=0.4):
        """
        Plot Greek sensitivity to underlying or time

        Parameters:
        -----------
        greek : str - 'delta', 'gamma', 'vega', 'theta', 'rho'
        variable : str - 'spot' or 'time'
        range_pct : float - Range to plot (e.g., 0.4 = ±40%)
        """
        greek = greek.lower()
        variable = variable.lower()

        if variable == 'spot':
            x_values = np.linspace(self.S * (1 - range_pct),
                                  self.S * (1 + range_pct), 100)
            y_values = []

            for S_val in x_values:
                pricer = OptionPricer(S_val, self.K, self.T, self.r,
                                    self.sigma, self.option_type)
                y_values.append(getattr(pricer, greek)())

            xlabel = 'Spot Price ($)'

        else:  # time
            x_values = np.linspace(0.01, self.T * 2, 100)
            y_values = []

            for T_val in x_values:
                pricer = OptionPricer(self.S, self.K, T_val, self.r,
                                    self.sigma, self.option_type)
                y_values.append(getattr(pricer, greek)())

            xlabel = 'Time to Maturity (Years)'

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, linewidth=2.5, color='steelblue')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.axvline(x=self.S if variable == 'spot' else self.T,
                   color='green', linestyle='--', alpha=0.5, linewidth=1,
                   label='Current Value')

        plt.xlabel(xlabel, fontsize=11)
        plt.ylabel(f'{greek.capitalize()}', fontsize=11)
        plt.title(f'{greek.capitalize()} Sensitivity - {self.option_type.capitalize()} Option',
                 fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_all_greeks_dashboard(self):
        """Create comprehensive dashboard with all Greeks"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Greeks Dashboard - {self.option_type.capitalize()} Option',
                    fontsize=16, fontweight='bold')

        spot_range = np.linspace(self.S * 0.6, self.S * 1.4, 100)
        greeks_list = ['delta', 'gamma', 'vega', 'theta', 'rho', 'price']
        colors = ['green', 'purple', 'orange', 'red', 'blue', 'steelblue']

        for idx, (greek, color) in enumerate(zip(greeks_list, colors)):
            ax = axes[idx // 3, idx % 3]

            y_values = []
            for S_val in spot_range:
                pricer = OptionPricer(S_val, self.K, self.T, self.r,
                                    self.sigma, self.option_type)
                y_values.append(getattr(pricer, greek)())

            ax.plot(spot_range, y_values, linewidth=2, color=color)
            ax.axvline(x=self.S, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
            ax.set_xlabel('Spot Price ($)', fontsize=9)
            ax.set_ylabel(f'{greek.capitalize()}', fontsize=9)
            ax.set_title(f'{greek.capitalize()}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.show()


def main():
    """Main execution - Example usage"""

    # Example 1: Basic usage
    print("\n" + "="*60)
    print("GREEK ANALYTICS & VOLATILITY SURFACE MONITOR")
    print("="*60)

    # Create analyzer
    analyzer = GreekAnalyzer(
        S=100,      # Spot price
        K=100,      # Strike price
        T=0.25,     # 3 months
        r=0.05,     # 5% risk-free rate
        sigma=0.25, # 25% volatility
        option_type='call'
    )

    # Print Greeks
    analyzer.print_greeks()

    # Plot volatility surface
    print("Generating volatility surface...")
    analyzer.plot_volatility_surface()

    # Plot individual Greek sensitivities
    print("Generating Delta sensitivity...")
    analyzer.plot_greek_sensitivity('delta', 'spot')

    print("Generating Gamma sensitivity...")
    analyzer.plot_greek_sensitivity('gamma', 'spot')

    print("Generating Theta decay...")
    analyzer.plot_greek_sensitivity('theta', 'time')

    # Complete dashboard
    print("Generating complete Greeks dashboard...")
    analyzer.plot_all_greeks_dashboard()

    # Example 2: Implied Volatility calculation
    print("\n" + "="*60)
    print("IMPLIED VOLATILITY CALCULATION")
    print("="*60)

    market_price = 5.50
    iv = OptionPricer.implied_volatility(
        S=100, K=100, T=0.25, r=0.05,
        market_price=market_price,
        option_type='call'
    )

    if iv:
        print(f"Market Price: ${market_price:.2f}")
        print(f"Implied Volatility: {iv*100:.2f}%")
    else:
        print("Could not calculate implied volatility")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
