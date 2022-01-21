from functions import *


def main(dim_kernel, sigma, std_dev, lambda_value, iteration, img_name):

    x0 = np.zeros((512, 512))

    ''' --- PUNTO 1 --- '''

    # Caricamento dell'immagine
    img = plt.imread(f"imgs/sample{img_name}.png").astype(np.float64)
    # Generazione del filtro di blur
    K = psf_fft(gaussian_kernel(dim_kernel, sigma), dim_kernel, x0.shape)
    # Generazione del rumore
    noise = np.random.normal(size = x0.shape) * std_dev
    # Applicazione del filtro di blur e del rumore
    blurred = A(img, K)
    noised = blurred + noise


    ''' --- PUNTO 2 --- '''

    # Funzioni da minimizzare calcolando una soluzione Naive
    def f_naive(x):
        X = x.reshape(512, 512)
        res = 0.5 * (np.linalg.norm(A(X, K) - noised)) ** 2
        return res

    def df_naive(x):
        X = x.reshape(512, 512)
        res = AT(A(X, K) - noised, K)
        res2 = np.reshape(res, 512 * 512)
        return res2

    # Ricostruzione dell'immagine usando il metodo del gradiente coniugato implementato dalla funzione minimize di libreria
    res = minimize(f_naive, x0, method='CG', jac=df_naive, options={'maxiter': 100})
    img_naive = res.x.reshape(512, 512)



    ''' --- PUNTO 3 --- '''

    # Funzioni da minimizzare con il termine di regolarizzazione di Tikhonov
    def f_reg(x):
        X = x.reshape(512, 512)
        res = 0.5 * (np.linalg.norm(A(X, K) - noised)) ** 2 + \
            (lambda_value / 2) * (np.linalg.norm(X) ** 2)
        return res

    def df_reg(x):
        X = x.reshape(512, 512)
        res = AT(A(X, K) - noised, K) + (lambda_value * X)
        res2 = np.reshape(res, 512*512)
        return res2

    # Ricostruzione dell'immagine utilizzando il metodo del gradiente coniugato, la funzione minimize di libreria
    # e il termine di regolarizzazione di Tikhonov
    res = minimize(f_reg, x0, method='CG', jac=df_reg, options={'maxiter': 100})
    img_reg = res.x.reshape(512, 512)

    # Ricostruzione dell'immagine utilizzando utilizzando il metodo del gradiente illustrato a lezione e il termine di regolarizzazione di Tikhonov
    (img_reg_2, norm_g_list, fun_eval_list, errors, iterations) = custom_minimize(x0, noised, 100, 1.e-5, f_reg, df_reg)



    ''' --- PUNTO 4 --- '''

    # Funzioni da minimizzare con Variazione Totale
    def f_totvar(x):
        X = x.reshape(512, 512)
        res = 0.5 * (np.linalg.norm(A(X, K) - noised)) ** 2 + (lambda_value * totvar(X))
        return res

    def df_totvar(x):
        X = x.reshape(512, 512)
        res = AT(A(X, K) - noised, K) + (lambda_value * grad_totvar(X))
        res2 = np.reshape(res, 512 * 512)
        return res2

    # Ricostruzione dell'immagine utilizzando il metodo del gradiente implementato a lezione
    # e come termine di regolarizzazione la funzione di "Variazione Totale"
    (img_totvar, norm_g_list_totvar, fun_eval_list_totvar, errors_totvar, iterations_totvar) = totvar_minimize(x0, noised, 100, 1.e-5, f_totvar, df_totvar)


    # Plotting dei grafici rappresentanti la variazione
    plt.plot(errors[1:])
    plt.plot(errors_totvar[1:])
    plt.legend(['Tikhonov', 'TotVar'])
    plt.xlabel('Iterazioni')
    plt.ylabel('Errore')
    plt.title('Iterazioni & Andamento dell\'errore')
    plt.show()
    plt.savefig('Tikh_TotVar_Geometrica.png', dpi=300)

    # Calcolo (e salvataggio su file) di PSNR e MSE
    # calc_PSNR_MSE(iteration, img, noised, img_naive, img_reg, img_reg_2, img_totvar, img_name)

    # Plotting delle immagini ottenute
    # plot_figure(img, noised, img_naive, img_reg, img_reg_2, img_totvar)


if __name__ == "__main__":
    # Utilizzando tale funzione ( main ) possiamo settare i valori sottostanti per effettuare tutti i test sulle immagine caricate.

    dim_kernel = [5, 7, 9]
    ker_sigma = [0.5, 1, 1.3]
    sigma = [0.01, 0.02, 0.03, 0.04, 0.05]
    lambda_value = [0.01, 0.05, 0.08, 0.32, 1]

    # Il booleano serve per attivare o meno i test sull'intero dataset; per eseguire test su una singola immagine del
    # dataset basta impostarlo a False.
    all_tests = False

    if all_tests:
        for img in range(10):
            # Creazione dei file che conterranno i PSNR e MSE delle immagini analizzate.
            output_PSNR = open(f"sample{img+1}PSNR.csv", 'w')
            output_PSNR.write(f"sample{img+1},Noised,Naive,Regolarized,Regolarized 2nd,TV correction\n")
            output_PSNR.close()
            output_MSE = open(f"sample{img+1}MSE.csv", 'w')
            output_MSE.write(f"sample{img+1},Noised,Naive,Regolarized,Regolarized 2nd,TV correction\n")
            output_MSE.close()
            for i in range(len(dim_kernel)):
                for j in range(len(sigma)):
                    for q in range(len(lambda_value)):
                        options = f"K{i + 1}_{sigma[j]}_{lambda_value[q]}"
                        main(dim_kernel[i], ker_sigma[i], sigma[j], lambda_value[q], options, img + 1)
    else:
        main(dim_kernel[0], ker_sigma[0], sigma[4], lambda_value[2], 'K1_0.05_0.08', '5')
