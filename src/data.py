"""
Data loading and cleaning for Online Retail II dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_online_retail_ii(file_2009_2010, file_2010_2011):
    """
    Load Online Retail II dataset from two files (supports CSV and Excel).
    """
    def read_file(filepath):
        ext = os.path.splitext(str(filepath))[1].lower()
        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use .csv, .xlsx, or .xls")
    
    print("Loading Year 2009-2010...")
    df1 = read_file(file_2009_2010)
    print(f"  Loaded {len(df1):,} rows")
    
    print("Loading Year 2010-2011...")
    df2 = read_file(file_2010_2011)
    print(f"  Loaded {len(df2):,} rows")
    
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"\nTotal rows: {len(df):,}")
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    column_mapping = {
        'Invoice': 'InvoiceNo',
        'Customer ID': 'CustomerID',
        'InvoiceDate': 'InvoiceDate',
        'StockCode': 'StockCode',
        'Description': 'Description',
        'Quantity': 'Quantity',
        'Price': 'Price',
        'Country': 'Country'
    }
    df = df.rename(columns=column_mapping)
    
    return df

def clean_data(df, verbose=True):
    """
    Clean Online Retail II data following standard preprocessing steps.
    
    Steps:
    1. Remove rows with missing CustomerID
    2. Remove cancellations (InvoiceNo starts with 'C')
    3. Filter Quantity > 0 and Price > 0
    4. Convert InvoiceDate to datetime
    5. Remove duplicates
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data
    verbose : bool
        Print cleaning statistics
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    if verbose:
        print("\n" + "="*60)
        print("DATA CLEANING PIPELINE")
        print("="*60)
        print(f"Initial rows: {len(df):,}")
    
    initial_rows = len(df)
    
    # Step 1: Remove missing CustomerID
    before = len(df)
    df = df[df['CustomerID'].notna()]
    if verbose:
        print(f"✓ Removed missing CustomerID: {before - len(df):,} rows removed")
    
    # Step 2: Remove cancellations (Invoice starts with 'C')
    before = len(df)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    if verbose:
        print(f"✓ Removed cancellations: {before - len(df):,} rows removed")
    
    # Step 3: Filter positive quantities and prices
    before = len(df)
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    if verbose:
        print(f"✓ Removed invalid Quantity/Price: {before - len(df):,} rows removed")
    
    # Step 4: Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    if verbose:
        print(f"✓ Converted InvoiceDate to datetime")
    
    # Step 5: Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    if verbose:
        print(f"✓ Removed duplicates: {before - len(df):,} rows removed")
    
    # Final statistics
    if verbose:
        print("="*60)
        print(f"Final rows: {len(df):,} ({len(df)/initial_rows*100:.1f}% retained)")
        print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
        print(f"Unique customers: {df['CustomerID'].nunique():,}")
        print(f"Unique invoices: {df['InvoiceNo'].nunique():,}")
        print("="*60 + "\n")
    
    return df


def get_dataset_stats(df):
    """
    Get comprehensive dataset statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned transaction data
        
    Returns:
    --------
    dict
        Dictionary with dataset statistics
    """
    stats = {
        'total_rows': len(df),
        'total_customers': df['CustomerID'].nunique(),
        'total_invoices': df['InvoiceNo'].nunique(),
        'date_range': (df['InvoiceDate'].min(), df['InvoiceDate'].max()),
        'countries': df['Country'].nunique(),
        'products': df['StockCode'].nunique(),
        'avg_quantity': df['Quantity'].mean(),
        'avg_price': df['Price'].mean(),
        'total_revenue': (df['Quantity'] * df['Price']).sum()
    }
    
    return stats


def print_dataset_info(df):
    """
    Print comprehensive dataset information.
    """
    stats = get_dataset_stats(df)
    
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Total Transactions (rows): {stats['total_rows']:,}")
    print(f"Unique Customers: {stats['total_customers']:,}")
    print(f"Unique Invoices: {stats['total_invoices']:,}")
    print(f"Date Range: {stats['date_range'][0].date()} to {stats['date_range'][1].date()}")
    print(f"Countries: {stats['countries']}")
    print(f"Unique Products: {stats['products']:,}")
    print(f"Average Quantity per Line: {stats['avg_quantity']:.2f}")
    print(f"Average Price per Item: ${stats['avg_price']:.2f}")
    print(f"Total Revenue: ${stats['total_revenue']:,.2f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Data loading and cleaning module")
    print("Use: from src.data import load_online_retail_ii, clean_data")
