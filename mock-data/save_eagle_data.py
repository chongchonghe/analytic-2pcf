import os
import numpy as np
import eagleSqlTools as sql
import csv

def main():
    sim_name = "RefL0050N0752_Subhalo"
    # sim_name = "RefL0100N1504_Subhalo"
    myQuery = """
SELECT
  SH.CentreOfPotential_x as sh_x,
  SH.CentreOfPotential_y as sh_y,
  SH.CentreOfPotential_z as sh_z
FROM
  {} as SH
WHERE
  SH.SnapNum = 27
    """.format(sim_name)
    fn_data = f"eagle_data_{sim_name}.npy"
    if not os.path.exists(fn_data):
        con = sql.connect("gpf042", password="KUZ863KL")
        myData = sql.execute_query(con, myQuery)
        data = np.vstack((myData["sh_x"], myData["sh_y"], myData["sh_z"]))
        # print(data.shape)
        # return
        with open(fn_data, "wb") as f:
            np.save(f, data)
            print(f"{fn_data} saved.")

    if sim_name == "RefL0050N0752_Subhalo":
        text = """L [cMpc], 50
N (total number of particles), 2x752^3 = 8.5052e+08
m_dm (dark matter particle mass), 9.70e6
        """
    elif sim_name == "RefL0100N1504_Subhalo":
        text = """L [cMpc], 100
N (total number of particles), 2x1504^3
m_dm (dark matter particle mass), 9.70e6
        """

    with open(f"eagle_data_{sim_name}.info", "w") as f:
        f.write(text)

    with open(f"eagle_data_{sim_name}.info", "r") as f:
        d = csv.reader(f)
        for l in d:
            print(f"Box size is {l[1]}")
            break

if __name__ == '__main__':
    main()
