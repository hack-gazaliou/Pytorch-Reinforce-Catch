
from engine import engine
from view import View
from VPGAgent import VPGAgent
import torch
from train_vpg import train_one_epoch     
import pandas as pd
import time

ALGO_CHOISIE = "VPG"
MAX_EPOCHS = 251
RECORD_EPOCHS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250]
history = {
    "loss": [],
    "avg_return": [],
    "std_return": [], 
    "avg_len": []     
}



if __name__ == "__main__":
    my_eng = engine()
    my_view = View(my_eng)
    
    match ALGO_CHOISIE :
        case "VPG":
            my_agent = VPGAgent()
            print("VPG agent initialised !")
        case "DQN":
            pass
        case "ES":
            pass
    
    starttime = time.time()
    for epoch in range (1, MAX_EPOCHS): 
        if epoch in RECORD_EPOCHS: #capture of keys epochs
            print(f" Enregistrement de l'epoch {epoch}...")
            obs = my_eng.reset()
            done = False
            frame_count = 0
            my_eng.lvl = 2 #rendering on medium level
            
            while not done: 
                my_view.render(my_eng)
                my_view.save_frame(epoch=epoch, frame_idx=frame_count)
                act = my_agent.get_action(torch.as_tensor(obs, dtype=torch.float32))
                obs, _, done = my_eng.step(act)
                frame_count += 1
                if frame_count > 1000: break #aucune raison d'arriver but we never know
            my_view.show_game_over(my_eng.score)
            for _ in range(10): 
                my_view.save_frame(epoch=epoch, frame_idx=frame_count)
                frame_count += 1
            time.sleep(3)
        loss, ret, std, length = train_one_epoch(my_agent, my_eng)

        history["loss"].append(loss)
        history["avg_return"].append(ret)
        history["std_return"].append(std)
        history["avg_len"].append(length)
        
        print(f"Epoch {epoch} | Loss: {loss:.3f} | Score Moyen: {ret:.1f} | Durée Moy: {length:.1f}")

    df = pd.DataFrame(history)
    df.to_csv("training_results.csv", index=False)
    print("Données sauvegardées dans training_results.csv")
    endtime = time.time()
    dt = int(endtime - starttime)
    hours = dt // 3600
    minutes = (dt%3600) //60
    sec = (dt%3600)%60
    print(f"Training time : {hours}h {minutes}min {sec}s")