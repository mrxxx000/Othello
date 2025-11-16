package main;

import com.eudycontreras.othello.capsules.AgentMove;
import com.eudycontreras.othello.capsules.ObjectiveWrapper;
import com.eudycontreras.othello.controllers.Agent;
import com.eudycontreras.othello.controllers.AgentController;
import com.eudycontreras.othello.enumerations.PlayerTurn;
import com.eudycontreras.othello.models.GameBoardState;

import java.util.List;

/**
 * AI Agent implementation using Minimax algorithm with Alpha-Beta Pruning.
 * 
 * This agent implements the adversarial search algorithm to find the optimal move
 * in the Othello game. It uses:
 * - Minimax algorithm for decision making
 * - Alpha-Beta pruning for optimization
 * - Depth-limited search with time constraints
 * - Heuristic evaluation function for non-terminal states
 * 
 * Assignment: α-β Pruning Minimax – Othello
 * Course: DA272E – Artificial Intelligence
 */
public class MyAgent extends Agent {
    
    // Maximum search depth (number of plies to look ahead)
    private static final int MAX_DEPTH = 6;
    
    // Time limit for a move in milliseconds (5 seconds)
    private static final long TIME_LIMIT = 5000;
    
    // Margin to ensure we don't exceed time limit
    private static final long TIME_MARGIN = 500;
    
    // Statistics tracking
    private int nodesExamined;
    private int searchDepth;
    private int reachedLeafNodes;
    private int prunedCounter;
    
    /**
     * Default constructor
     */
    public MyAgent() {
        super(PlayerTurn.PLAYER_ONE);
        this.agentName = "Alpha-Beta AI";
    }
    
    /**
     * Constructor with player turn
     * @param playerTurn The player turn (PLAYER_ONE or PLAYER_TWO)
     */
    public MyAgent(PlayerTurn playerTurn) {
        super(playerTurn);
        this.agentName = "Alpha-Beta AI";
    }
    
    /**
     * Constructor with name and player turn
     * @param name The agent's name (max 14 characters)
     * @param playerTurn The player turn (PLAYER_ONE or PLAYER_TWO)
     */
    public MyAgent(String name, PlayerTurn playerTurn) {
        super(name, playerTurn);
    }

    /**
     * Main method to get the best move using Alpha-Beta Pruning Minimax algorithm.
     * This method is called by the game framework when it's the agent's turn.
     * 
     * @param gameState The current state of the game board
     * @return The best move found by the algorithm
     */
    @Override
    public AgentMove getMove(GameBoardState gameState) {
        // Reset performance counters
        resetCounters();
        
        // Record start time to enforce time limit
        long startTime = System.currentTimeMillis();
        
        // Get all available moves for the current player
        List<ObjectiveWrapper> availableMoves = AgentController.getAvailableMoves(gameState, playerTurn);
        
        // If no moves available, return invalid move
        if (availableMoves.isEmpty()) {
            System.out.println("No available moves!");
            return new MyAgentMove();
        }
        
        // If only one move available, return it immediately
        if (availableMoves.size() == 1) {
            System.out.println("Only one move available, returning it.");
            MyAgentMove move = new MyAgentMove(availableMoves.get(0).getObjectiveCell().getIndex(), 0);
            return move;
        }
        
        // Initialize best move tracking
        ObjectiveWrapper bestMove = null;
        double bestValue = Double.NEGATIVE_INFINITY;
        
        // Alpha-Beta initial values
        double alpha = Double.NEGATIVE_INFINITY;
        double beta = Double.POSITIVE_INFINITY;
        
        // Evaluate each possible move
        System.out.println("\n=== Alpha-Beta Search Started ===");
        System.out.println("Available moves: " + availableMoves.size());
        
        for (ObjectiveWrapper move : availableMoves) {
            // Create a new game state with this move applied
            GameBoardState childState = AgentController.getNewState(gameState, move);
            
            // Call minValue because opponent plays next
            double value = minValue(childState, alpha, beta, 1, startTime);
            
            System.out.println("Move [" + move.getObjectiveCell().getIndex().getRow() + "," + 
                             move.getObjectiveCell().getIndex().getCol() + "] Value: " + value);
            
            // Update best move if this move is better
            if (value > bestValue) {
                bestValue = value;
                bestMove = move;
            }
            
            // Update alpha for pruning
            alpha = Math.max(alpha, bestValue);
        }
        
        // Print search statistics and board status
        long elapsedTime = System.currentTimeMillis() - startTime;
        printMoveStatistics(gameState, elapsedTime, bestValue);
        
        // Create and return the move
        MyAgentMove agentMove = new MyAgentMove();
        if (bestMove != null) {
            agentMove.setMove(bestMove.getObjectiveCell().getIndex());
            agentMove.setValue(bestValue);
        }
        
        return agentMove;
    }
    
    /**
     * Maximizing player (AI agent) - tries to maximize the evaluation
     * 
     * @param state Current game state
     * @param alpha Best value maximizer can guarantee
     * @param beta Best value minimizer can guarantee
     * @param depth Current depth in the search tree
     * @param startTime Start time of the search
     * @return The maximum value achievable from this state
     */
    private double maxValue(GameBoardState state, double alpha, double beta, int depth, long startTime) {
        // Increment nodes examined counter
        nodesExamined++;
        
        // Check if we should cut off the search
        if (cutoffTest(state, depth, startTime)) {
            // Update maximum depth reached
            searchDepth = Math.max(searchDepth, depth);
            
            // Check if this is a terminal state (game over)
            if (state.isTerminal()) {
                reachedLeafNodes++;
            }
            
            // Return heuristic evaluation of this state
            return evaluateState(state);
        }
        
        // Initialize value to worst case for maximizer
        double value = Double.NEGATIVE_INFINITY;
        
        // Get all available moves for the maximizing player (AI)
        List<ObjectiveWrapper> moves = AgentController.getAvailableMoves(state, playerTurn);
        
        // If no moves available, pass turn to opponent
        if (moves.isEmpty()) {
            PlayerTurn opponent = getOpponent(playerTurn);
            List<ObjectiveWrapper> opponentMoves = AgentController.getAvailableMoves(state, opponent);
            
            if (opponentMoves.isEmpty()) {
                // Game over - both players can't move
                searchDepth = Math.max(searchDepth, depth);
                reachedLeafNodes++;
                return evaluateState(state);
            } else {
                // Pass turn to opponent
                return minValue(state, alpha, beta, depth + 1, startTime);
            }
        }
        
        // Evaluate each possible move
        for (ObjectiveWrapper move : moves) {
            // Create child state
            GameBoardState childState = AgentController.getNewState(state, move);
            
            // Recursively call minValue for opponent's turn
            value = Math.max(value, minValue(childState, alpha, beta, depth + 1, startTime));
            
            // Alpha-Beta pruning: if value >= beta, prune remaining branches
            if (value >= beta) {
                prunedCounter++;
                return value; // Beta cutoff
            }
            
            // Update alpha
            alpha = Math.max(alpha, value);
        }
        
        return value;
    }
    
    /**
     * Minimizing player (opponent) - tries to minimize the evaluation
     * 
     * @param state Current game state
     * @param alpha Best value maximizer can guarantee
     * @param beta Best value minimizer can guarantee
     * @param depth Current depth in the search tree
     * @param startTime Start time of the search
     * @return The minimum value achievable from this state
     */
    private double minValue(GameBoardState state, double alpha, double beta, int depth, long startTime) {
        // Increment nodes examined counter
        nodesExamined++;
        
        // Check if we should cut off the search
        if (cutoffTest(state, depth, startTime)) {
            // Update maximum depth reached
            searchDepth = Math.max(searchDepth, depth);
            
            // Check if this is a terminal state (game over)
            if (state.isTerminal()) {
                reachedLeafNodes++;
            }
            
            // Return heuristic evaluation of this state
            return evaluateState(state);
        }
        
        // Initialize value to worst case for minimizer
        double value = Double.POSITIVE_INFINITY;
        
        // Get opponent player
        PlayerTurn opponent = getOpponent(playerTurn);
        
        // Get all available moves for the minimizing player (opponent)
        List<ObjectiveWrapper> moves = AgentController.getAvailableMoves(state, opponent);
        
        // If no moves available, pass turn back to maximizer
        if (moves.isEmpty()) {
            List<ObjectiveWrapper> ourMoves = AgentController.getAvailableMoves(state, playerTurn);
            
            if (ourMoves.isEmpty()) {
                // Game over - both players can't move
                searchDepth = Math.max(searchDepth, depth);
                reachedLeafNodes++;
                return evaluateState(state);
            } else {
                // Pass turn back to maximizer
                return maxValue(state, alpha, beta, depth + 1, startTime);
            }
        }
        
        // Evaluate each possible move
        for (ObjectiveWrapper move : moves) {
            // Create child state
            GameBoardState childState = AgentController.getNewState(state, move);
            
            // Recursively call maxValue for AI's turn
            value = Math.min(value, maxValue(childState, alpha, beta, depth + 1, startTime));
            
            // Alpha-Beta pruning: if value <= alpha, prune remaining branches
            if (value <= alpha) {
                prunedCounter++;
                return value; // Alpha cutoff
            }
            
            // Update beta
            beta = Math.min(beta, value);
        }
        
        return value;
    }
    
    /**
     * Cutoff test to determine if we should stop searching deeper.
     * 
     * The search is cut off when:
     * 1. The state is terminal (game over)
     * 2. Maximum depth is reached
     * 3. Time limit is approaching
     * 
     * @param state Current game state
     * @param depth Current depth in search tree
     * @param startTime Start time of the search
     * @return true if search should be cut off, false otherwise
     */
    private boolean cutoffTest(GameBoardState state, int depth, long startTime) {
        // Check if terminal state (game over)
        if (state.isTerminal()) {
            return true;
        }
        
        // Check if maximum depth reached
        if (depth >= MAX_DEPTH) {
            return true;
        }
        
        // Check if time limit is approaching
        long elapsedTime = System.currentTimeMillis() - startTime;
        if (elapsedTime >= (TIME_LIMIT - TIME_MARGIN)) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Heuristic evaluation function for non-terminal states.
     * 
     * This function estimates the utility of a game state from the AI's perspective.
     * It uses the framework's built-in evaluation which considers:
     * - Disc count (number of pieces on the board)
     * - Mobility (number of available moves)
     * - Position weights (corners and edges are valuable)
     * 
     * @param state The game state to evaluate
     * @return A numerical value representing the utility of the state
     */
    private double evaluateState(GameBoardState state) {
        // Use the framework's evaluation function
        return AgentController.getGameEvaluation(state, playerTurn);
    }
    
    /**
     * Get the opponent's player turn
     * 
     * @param player Current player
     * @return The opponent player
     */
    private PlayerTurn getOpponent(PlayerTurn player) {
        return player == PlayerTurn.PLAYER_ONE ? PlayerTurn.PLAYER_TWO : PlayerTurn.PLAYER_ONE;
    }
    
    /**
     * Reset performance counters before each move
     */
    @Override
    public void resetCounters() {
        super.resetCounters();
        nodesExamined = 0;
        searchDepth = 0;
        reachedLeafNodes = 0;
        prunedCounter = 0;
    }
    
    /**
     * Prints statistics after each move as required by assignment:
     * - Current board status
     * - Search depth
     * - Number of nodes examined
     * 
     * @param gameState Current game state
     * @param elapsedTime Time taken for the move
     * @param bestValue Best value found
     */
    private void printMoveStatistics(GameBoardState gameState, long elapsedTime, double bestValue) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("MOVE STATISTICS - " + this.getAgentName());
        System.out.println("=".repeat(70));
        
        // Print current board status
        System.out.println("\nCURRENT BOARD STATUS:");
        AgentManager.printBoard(gameState.getCells(), true);
        
        System.out.println("White pieces: " + gameState.getWhiteCount());
        System.out.println("Black pieces: " + gameState.getBlackCount());
        System.out.println("Total pieces: " + gameState.getTotalCount());        
        // Print search depth
        System.out.println("\nSEARCH DEPTH:");
        System.out.println("  Maximum depth limit: " + MAX_DEPTH);
        System.out.println("  Actual depth reached: " + searchDepth);
        
        // Print nodes examined
        System.out.println("\nNODES EXAMINED:");
        System.out.println("  Total nodes examined: " + nodesExamined);
        System.out.println("  Leaf nodes reached: " + reachedLeafNodes);
        System.out.println("  Branches pruned: " + prunedCounter);
        
        System.out.println("\nADDITIONAL INFO:");
        System.out.println("  Time taken: " + elapsedTime + " ms");
        System.out.println("  Best move value: " + bestValue);
        if (elapsedTime > 0) {
            System.out.println("  Nodes per second: " + (nodesExamined * 1000 / elapsedTime));
        }
        
        System.out.println("=".repeat(70) + "\n");
    }
}
