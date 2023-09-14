if __name__ == "__main__":
    # for local dev
    import os
    import sys
    sys.path.append(os.getcwd())
    from plugin import register
    register()
