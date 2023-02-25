import java.util.Arrays;
import java.util.PriorityQueue;

/**
 * Greedy Randomized Adaptative Search Procedure
 */
public class GRASP implements Solver {
    final TSP tsp;
    final int ite;
    final int k;
    final boolean useRolette;
    private int runTime;

    public Route getBestSol() {
        return bestSol;
    }

    @Override
    public String toString() {
        return "GRASP{" +
                "ite=" + ite +
                ", k = " + k +
                ", runTime=" + runTime +
                ", bestSol=" + bestSol.cost +
                '}';
    }

    @Override
    public String parametros() {
        return String.join(",","ite=" + ite,
                "k=" + k);
    }

    @Override
    public int getRunTime() {
        return runTime;
    }

    private Route bestSol;

    public GRASP(TSP tsp, int ite, int k, boolean useRolette) {
        this.ite = ite;
        this.tsp = tsp;
        this.k = k;
        this.useRolette = useRolette;
        w = new double[k];
    }

    public void run() {
        long t = System.currentTimeMillis();
        Route currentSol = new Route(tsp);
        bestSol = new Route(tsp);
        bestSol.cost = Double.POSITIVE_INFINITY;
        VND vnd = new VND(tsp);
        for (int i = 0; i < ite; i++) {
            greedyRandom(currentSol);
            vnd.run(currentSol);
            if (currentSol.cost < bestSol.cost - Utils.EPS) {
                bestSol.copy(currentSol);
                System.out.println(i + " GRASP " + bestSol.cost);
                assert Utils.equals(bestSol.cost, tsp.cost(bestSol.v)) : "variável 'cost' está inconsistente";
            }
        }
        runTime = (int) (System.currentTimeMillis() - t);
    }

    private PriorityQueue<Candidate> fila = new PriorityQueue<>();

    private double w[];

    public void greedyRandom(Route currentSol) {
        int v[] = currentSol.v;
        boolean[] visitado = new boolean[tsp.N];
        v[0] = 0;
        visitado[0] = true;
        for (int i = 1; i < tsp.N; i++) {

            int arg_j = -1;
            fila.clear();
            for (int j = 0; j < tsp.N; j++)
                if (!visitado[j]) {
                    fila.add(new Candidate(j, tsp.c[v[i - 1]][j]));
                    if (fila.size() > this.k) {
                        fila.poll();
                    }
                }

            if (useRolette) {
                //sorteio por peso
                Object lista[] = fila.toArray();
                Arrays.fill(w, 0);
                for (int j = 0; j < lista.length; j++)
                    w[j] = 1 / (((Candidate) lista[j]).score + 1);
                int x = Utils.roulette(w);
                arg_j = ((Candidate) lista[x]).id;
            } else {
                // sorteio uniforme
                int x = Utils.rd.nextInt(fila.size());
                arg_j = ((Candidate) fila.toArray()[x]).id;
            }


            v[i] = arg_j;
            visitado[arg_j] = true;
        }
        Utils.rollZero(v);
        currentSol.cost = tsp.cost(v);
    }

    class Candidate implements Comparable<Candidate> {
        int id;
        double score;

        public Candidate(int id, double score) {
            this.id = id;
            this.score = score;
        }

        @Override
        public int compareTo(Candidate o) {
            return -Double.compare(score, o.score);
        }
    }
}
