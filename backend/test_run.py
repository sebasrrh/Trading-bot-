import logging
logging.basicConfig(level=logging.INFO)

print("Starting import...")
try:
    from main import perform_analysis
    from data.ingestion import get_all_data
except Exception as e:
    print("Import error:", e)

try:
    print("Testing get_all_data('AAPL')...")
    df, s, f = get_all_data("AAPL")
    print(f"Data fetched! DF size: {len(df)}")
    
    print("Testing perform_analysis()...")
    state = perform_analysis("AAPL")
    print("Success! state:", state['signal'])
except Exception as e:
    print("ERROR:", repr(e))
    import traceback
    traceback.print_exc()
