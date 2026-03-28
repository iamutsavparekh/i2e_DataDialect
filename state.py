import solara

class State:
    # Core Application State
    history = solara.reactive([])
    user_query = solara.reactive("")
    is_loading = solara.reactive(False)
    
    # Configuration State
    exec_mode = solara.reactive(True)  # True = Local, False = Cloud
    api_key = solara.reactive("")
    selected_model = solara.reactive("llama3.1:8b")
    
    # Theme State
    dark_mode = solara.reactive(True)