CONFIG["OPTUNA_CALISTIR"] = True
CONFIG["OPTUNA_TRIALS"] = 60
SECIMLER_GRID = dict(SECIMLER)          # grid seçimini yedekle
grid_yedek = grid.copy()


print("Grid seçimi   :", list(SECIMLER_GRID))
print("Optuna sonrası:", list(SECIMLER))
display(grid[grid["MODEL"].isin(["Huber", "SVR", "ExtraTrees", "GBoost", "Ridge*", "Lasso*", "ElasticNet*"])]
        [goster].head(10).style.format(fmt, decimal=","))
