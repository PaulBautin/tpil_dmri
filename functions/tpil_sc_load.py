def load_connectivity():
    # initialize A matrix which is the structural connectivity matrix NxN with N the number of nodes (n_nodes)
    A_con_v1 = np.load("/home/pabaua/Downloads/df_con_scilpy_mask_v1.npy")
    A_con_v2 = np.load("/home/pabaua/Downloads/df_con_scilpy_mask_v2.npy")
    A_con_v3 = np.load("/home/pabaua/Downloads/df_con_scilpy_mask_v3.npy")

    A_con_v1 = np.array(np.vsplit(A_con_v1, 24))[:, :210, :210]
    A_con_v2 = np.array(np.vsplit(A_con_v2, 25))[:, :210, :210]
    A_con_v3 = np.array(np.vsplit(A_con_v3, 25))[:, :210, :210]



    A_clbp_v1 = np.load("/home/pabaua/Downloads/df_clbp_scilpy_mask_v1.npy")
    A_clbp_v2 = np.load("/home/pabaua/Downloads/df_clbp_scilpy_mask_v2.npy")
    A_clbp_v3 = np.load("/home/pabaua/Downloads/df_clbp_scilpy_mask_v3.npy")

    A_clbp_v1 = np.array(np.vsplit(A_clbp_v1, 27))[:, :210, :210]
    A_clbp_v2 = np.array(np.vsplit(A_clbp_v2, 25))[:, :210, :210]
    A_clbp_v3 = np.array(np.vsplit(A_clbp_v3, 23))[:, :210, :210]

    A = {'v1':[A_con_v1, A_clbp_v1], 'v2':[A_con_v2, A_clbp_v2], 'v3':[A_con_v3, A_clbp_v3]}
    return A