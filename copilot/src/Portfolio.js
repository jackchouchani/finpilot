import React, { useState, useEffect, useCallback } from 'react';
import {
    TextField, Button, Table, TableBody, TableCell, TableHead, TableRow, Paper, List, ListItem, ListItemText, Typography, Dialog, DialogTitle, DialogContent, DialogActions, Select, MenuItem, CircularProgress, InputAdornment, Autocomplete, IconButton, Box, Switch, FormControlLabel, Tooltip
} from '@mui/material';
import {
    PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip as RechartsTooltip, LineChart, Line, XAxis, YAxis, CartesianGrid
} from 'recharts';
import SaveIcon from '@mui/icons-material/Save';
import SimulationIcon from '@mui/icons-material/Timeline';
import ReportIcon from '@mui/icons-material/Assessment';
import { Info } from '@mui/icons-material';
import axios from 'axios';

export const PortfolioPieChart = ({ portfolio }) => {
    if (!portfolio || !portfolio.stocks || portfolio.stocks.length === 0) {
        return <Typography>Aucune donnée de portefeuille disponible pour le graphique</Typography>;
    }

    const data = portfolio.stocks.map(stock => ({
        name: stock.symbol,
        value: parseFloat(stock.weight),
        entryPrice: parseFloat(stock.entry_price)
    }));

    const COLORS = ['#1976d2', '#2196f3', '#64b5f6', '#0d47a1', '#bbdefb'];

    const renderCustomizedLabel = ({
        cx, cy, midAngle, innerRadius, outerRadius, percent, index
    }) => {
        const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
        const x = cx + radius * Math.cos(-midAngle * Math.PI / 180);
        const y = cy + radius * Math.sin(-midAngle * Math.PI / 180);

        return (
            <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central">
                {`${data[index].name} ${(percent * 100).toFixed(0)}%`}
            </text>
        );
    };

    return (
        <ResponsiveContainer width="100%" height={400}>
            <PieChart>
                <Pie
                    data={data}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={renderCustomizedLabel}
                    outerRadius={150}
                    fill="#8884d8"
                    dataKey="value"
                >
                    {data.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                </Pie>
                <RechartsTooltip content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                            <div style={{ backgroundColor: '#fff', padding: '5px', border: '1px solid #ccc' }}>
                                <p>{`${data.name} : ${(data.value * 100).toFixed(2)}%`}</p>
                                <p>{`Prix d'entrée : $${data.entryPrice}`}</p>
                            </div>
                        );
                    }
                    return null;
                }} />
                <Legend />
            </PieChart>
        </ResponsiveContainer>
    );
};

function Portfolio() {
    const [portfolio, setPortfolio] = useState({ name: 'default', stocks: [] });
    const [newStock, setNewStock] = useState({ symbol: '', weight: '', entryPrice: '' });
    const [livePrices, setLivePrices] = useState({});
    const [news, setNews] = useState([]);
    const [scenarioResults, setScenarioResults] = useState(null);
    const [openScenario, setOpenScenario] = useState(false);
    const [selectedScenario, setSelectedScenario] = useState('market_crash');
    const [openReport, setOpenReport] = useState(false);
    const [reportData, setReportData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [portfolioValue, setPortfolioValue] = useState(100000); // Valeur par défaut
    const [displayMode, setDisplayMode] = useState('weight'); // 'weight' ou 'shares'
    const [editingStock, setEditingStock] = useState(null);

    const handleSubmit = (event) => {
        event.preventDefault();
        if (newStock.symbol && newStock.weight && newStock.entryPrice) {
            addStock();
        } else {
            alert('Veuillez remplir tous les champs');
        }
    };

    const handleChange = (e) => {
        setNewStock({ ...newStock, [e.target.name]: e.target.value });
    };

    useEffect(() => {
        fetchPortfolio();
    }, []);

    const fetchLatestPrice = useCallback(async (symbol) => {
        try {
            const response = await axios.get(`${process.env.REACT_APP_API_URL}/latest_price?symbol=${symbol}`);
            return response.data.price;
        } catch (error) {
            console.error(`Erreur lors de la récupération du prix pour ${symbol}:`, error);
            return null;
        }
    }, []);

    const updateAllPrices = useCallback(async () => {
        if (!portfolio.stocks || portfolio.stocks.length === 0) return;

        const updatedPrices = { ...livePrices };
        for (const stock of portfolio.stocks) {
            const price = await fetchLatestPrice(stock.symbol);
            if (price !== null) {
                updatedPrices[stock.symbol] = price;
            }
        }
        setLivePrices(updatedPrices);
    }, [portfolio.stocks, fetchLatestPrice, livePrices]);


    useEffect(() => {
        updateAllPrices();
        // Vous pouvez également ajouter un intervalle pour mettre à jour les prix régulièrement si nécessaire
        // const interval = setInterval(fetchLatestPrices, 60000); // Mise à jour toutes les minutes
        // return () => clearInterval(interval);
    }, [updateAllPrices]);

    useEffect(() => {
        const fetchNewsAndTranslate = async () => {
            if (!portfolio || !portfolio.stocks) {
                console.error("Portfolio ou portfolio.stocks est indéfini");
                return;
            }
            const tickers = portfolio.stocks.map(stock => stock.symbol).join(',');
            try {
                const response = await axios.get(`${process.env.REACT_APP_API_URL}/news?tickers=${tickers}`);
                console.log("Nouvelles reçues:", response.data);
                const translatedResponse = await axios.post(`${process.env.REACT_APP_API_URL}/translate_news`, {
                    news: response.data
                }, {
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                console.log("Nouvelles traduites:", translatedResponse.data);
                setNews(translatedResponse.data);
            } catch (error) {
                console.error("Erreur lors de la récupération ou de la traduction des nouvelles:", error);
            } finally {
                setLoading(false);
            }
        };
        fetchNewsAndTranslate();
    }, [portfolio]);


    useEffect(() => {
        if (newStock.symbol) {
            const fetchNewStockPrice = async () => {
                try {
                    const response = await axios.get(`${process.env.REACT_APP_API_URL}/latest_price?symbol=${newStock.symbol}`);
                    if (response.data && response.data.price) {
                        setLivePrices(prev => ({ ...prev, [newStock.symbol]: response.data.price }));
                    }
                } catch (error) {
                    console.error(`Erreur lors de la récupération du prix pour ${newStock.symbol}:`, error);
                }
            };
            fetchNewStockPrice();
        }
    }, [newStock.symbol]);

    useEffect(() => {
        const fetchPortfolioValue = async () => {
            try {
                const response = await axios.get(process.env.REACT_APP_API_URL + '/get_portfolio_value', {
                    headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
                });
                setPortfolioValue(response.data.portfolio_value);
            } catch (error) {
                console.error("Error fetching portfolio value:", error);
            }
        };
        fetchPortfolioValue();
    }, []);

    const updatePortfolioValue = async (newValue) => {
        try {
            await axios.post(process.env.REACT_APP_API_URL + '/update_portfolio_value',
                { portfolio_value: newValue },
                { headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` } }
            );
            setPortfolioValue(newValue);
        } catch (error) {
            console.error("Error updating portfolio value:", error);
        }
    };

    const fetchPortfolio = async () => {
        try {
            const response = await axios.get(process.env.REACT_APP_API_URL + '/portfolio');
            setPortfolio(response.data);
        } catch (error) {
            console.error("Erreur lors de la récupération du portfolio:", error);
        } finally {
            setLoading(false);
        }
    };

    const calculateValue = (stock) => {
        const currentPrice = livePrices[stock.symbol] || parseFloat(stock.entry_price);
        return displayMode === 'weight'
            ? (parseFloat(stock.weight) / 100) * portfolioValue
            : parseFloat(stock.shares || stock.weight) * currentPrice;
    };

    const addStock = () => {
        setPortfolio(prevPortfolio => ({
            ...prevPortfolio,
            stocks: [...(prevPortfolio.stocks || []), {
                symbol: newStock.symbol,
                weight: newStock.weight,
                entry_price: newStock.entryPrice
            }]
        }));
        setNewStock({ symbol: '', weight: '', entryPrice: '' });
    };

    const savePortfolio = async () => {
        try {
            const response = await axios.post(process.env.REACT_APP_API_URL + '/portfolio', portfolio);
            if (response.status === 200) {
                alert('Portfolio sauvegardé avec succès!');
                fetchPortfolio();
            } else {
                alert('Erreur lors de la sauvegarde du portfolio.');
            }
        } catch (error) {
            console.error("Erreur lors de la sauvegarde du portfolio:", error);
            alert('Erreur lors de la sauvegarde du portfolio. Veuillez réessayer.');
        }
    };

    const simulateScenario = async () => {
        try {
            const response = await axios.post(process.env.REACT_APP_API_URL + '/simulate_scenario', {
                portfolio: {
                    stocks: portfolio.stocks.map(stock => ({
                        symbol: stock.symbol,
                        weight: parseFloat(stock.weight)
                    }))
                },
                scenario: selectedScenario
            });
            setScenarioResults(response.data);
            setOpenScenario(true);
        } catch (error) {
            console.error("Erreur lors de la simulation du scénario:", error);
            alert('Erreur lors de la simulation du scénario. Veuillez réessayer.');
        }
    };

    const generateReport = async () => {
        setLoading(true);
        try {
            const response = await axios.post(
                `${process.env.REACT_APP_API_URL}/generate_report`,
                { portfolio: portfolio },
                {
                    responseType: 'json',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                }
            );

            console.log("Response data:", response.data); // Ajoutez cette ligne pour déboguer

            if (response.data && response.data.report) {
                setReportData(response.data.report);
                setOpenReport(true);
            } else {
                console.error("Invalid report data:", response.data); // Ajoutez cette ligne
                throw new Error('Invalid report data received');
            }
        } catch (error) {
            console.error("Erreur lors de la génération du rapport:", error);
            alert(`Erreur lors de la génération du rapport: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleEditStock = (index) => {
        setEditingStock(index);
    };

    const handleSaveStock = (index) => {
        setEditingStock(null);
        // Ici, vous pouvez ajouter une logique pour sauvegarder les modifications dans la base de données
    };

    const handleStockChange = (index, field, value) => {
        const updatedStocks = [...portfolio.stocks];
        updatedStocks[index] = { ...updatedStocks[index], [field]: value };
        setPortfolio({ ...portfolio, stocks: updatedStocks });
    };

    const formatNumber = (number) => {
        return new Intl.NumberFormat('fr-FR', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(number);
    };

    const calculateDiff = (stock) => {
        const entryPrice = parseFloat(stock.entry_price);
        const currentPrice = livePrices[stock.symbol] || entryPrice;
        const shares = displayMode === 'weight'
            ? (parseFloat(stock.weight) / 100) * portfolioValue / entryPrice
            : parseFloat(stock.weight);
        const diff = (currentPrice - entryPrice) * shares;
        const color = diff >= 0 ? 'green' : 'red';
        return { diff, color };
    };

    if (loading) {
        return <CircularProgress />;
    }

    return (
        <Paper sx={{ padding: 2 }}>
            <Typography variant="h6">Dernières Nouvelles</Typography>
            {loading ? (
                <CircularProgress />
            ) : news && news.length > 0 ? (
                <List>
                    {news.map((item, index) => (
                        <ListItem key={index}>
                            <ListItemText primary={item.title} secondary={item.description} />
                        </ListItem>
                    ))}
                </List>
            ) : (
                <Typography>Aucune nouvelle disponible pour les actions actuelles</Typography>
            )}

            <form onSubmit={handleSubmit}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                    <Autocomplete
                        options={['AAPL', 'GOOGL', 'MSFT', 'AMZN']}
                        renderInput={(params) => <TextField {...params} label="Symbole de l'action" />}
                        onInputChange={(event, newValue) => setNewStock({ ...newStock, symbol: newValue })}
                        value={newStock.symbol}
                        sx={{ flexGrow: 1 }}
                    />
                    <TextField
                        label={displayMode === 'weight' ? "Poids (%)" : "Nombre d'actions"}
                        type="number"
                        name={displayMode === 'weight' ? 'weight' : 'shares'}
                        value={displayMode === 'weight' ? newStock.weight : newStock.shares}
                        onChange={handleChange}
                        InputProps={{
                            endAdornment: displayMode === 'weight' ? <InputAdornment position="end">%</InputAdornment> : null,
                        }}
                        sx={{ flexGrow: 1 }}
                    />
                    <TextField
                        label="Prix d'entrée"
                        type="number"
                        name="entryPrice"
                        value={newStock.entryPrice}
                        onChange={handleChange}
                        sx={{ flexGrow: 1 }}
                    />
                    <Tooltip title={`Le ${displayMode === 'weight' ? 'poids' : 'nombre d\'actions'} représente ${displayMode === 'weight' ? 'le pourcentage' : 'la quantité'} de l'action dans votre portefeuille.`}>
                        <IconButton>
                            <Info />
                        </IconButton>
                    </Tooltip>
                    <Button type="submit" variant="contained" color="primary">Ajouter</Button>
                </Box>
            </form>

            {portfolio && portfolio.stocks && portfolio.stocks.length > 0 ? (
                <>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Symbole</TableCell>
                                <TableCell>{displayMode === 'weight' ? 'Poids (%)' : 'Actions'}</TableCell>
                                <TableCell>Prix d'entrée</TableCell>
                                <TableCell>Prix actuel</TableCell>
                                <TableCell>Valeur</TableCell>
                                <TableCell>Différence</TableCell>
                                <TableCell>Actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {portfolio.stocks.map((stock, index) => {
                                const currentPrice = livePrices[stock.symbol] || parseFloat(stock.entry_price);
                                const shares = displayMode === 'weight'
                                    ? (parseFloat(stock.weight) / 100) * portfolioValue / parseFloat(stock.entry_price)
                                    : parseFloat(stock.weight);
                                const value = shares * currentPrice;
                                const { diff, color } = calculateDiff(stock);
                                return (
                                    <TableRow key={index}>
                                        <TableCell>{stock.symbol}</TableCell>
                                        <TableCell>
                                            {editingStock === index ? (
                                                <TextField
                                                    type="number"
                                                    value={displayMode === 'weight' ? stock.weight : shares}
                                                    onChange={(e) => handleStockChange(index, displayMode === 'weight' ? 'weight' : 'shares', e.target.value)}
                                                    InputProps={displayMode === 'weight' ? {
                                                        endAdornment: <InputAdornment position="end">%</InputAdornment>,
                                                    } : {}}
                                                />
                                            ) : (
                                                displayMode === 'weight'
                                                    ? `${formatNumber(parseFloat(stock.weight))}%`
                                                    : formatNumber(shares)
                                            )}
                                        </TableCell>
                                        <TableCell>
                                            {editingStock === index ? (
                                                <TextField
                                                    type="number"
                                                    value={stock.entry_price}
                                                    onChange={(e) => handleStockChange(index, 'entry_price', e.target.value)}
                                                />
                                            ) : (
                                                formatNumber(parseFloat(stock.entry_price))
                                            )}
                                        </TableCell>
                                        <TableCell>
                                            {livePrices[stock.symbol]
                                                ? formatNumber(livePrices[stock.symbol])
                                                : 'Chargement...'}
                                        </TableCell>
                                        <TableCell>{formatNumber(value)}</TableCell>
                                        <TableCell style={{ color }}>{formatNumber(diff)}</TableCell>
                                        <TableCell>
                                            {editingStock === index ? (
                                                <Button onClick={() => handleSaveStock(index)}>Save</Button>
                                            ) : (
                                                <Button onClick={() => handleEditStock(index)}>Edit</Button>
                                            )}
                                        </TableCell>
                                    </TableRow>
                                );
                            })}
                        </TableBody>
                    </Table>
                    <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>Allocation du Portefeuille</Typography>
                    <PortfolioPieChart portfolio={portfolio} />
                </>
            ) : (
                <Typography sx={{ my: 2 }}>Le portefeuille est actuellement vide. Ajoutez des actions pour commencer.</Typography>
            )}

            <Box sx={{ mt: 3 }}>
                <TextField
                    label="Valeur du Portefeuille"
                    value={formatNumber(portfolioValue)}
                    onChange={(e) => {
                        const newValue = parseFloat(e.target.value.replace(/[^0-9,-]/g, '').replace(',', '.'));
                        if (!isNaN(newValue)) {
                            setPortfolioValue(newValue);
                            updatePortfolioValue(newValue);
                        }
                    }}
                    fullWidth
                    margin="normal"
                />
                <Typography variant="h6" sx={{ mt: 2 }}>Valeur Actuelle du Portefeuille: {portfolioValue.toFixed(2)}</Typography>
                <FormControlLabel
                    control={<Switch checked={displayMode === 'shares'} onChange={() => setDisplayMode(displayMode === 'weight' ? 'shares' : 'weight')} />}
                    label={`Passer en mode ${displayMode === 'weight' ? 'Actions' : 'Poids'}`}
                    sx={{ mt: 2 }}
                />
            </Box>

            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button onClick={() => savePortfolio(portfolio.stocks)} variant="contained" color="secondary" startIcon={<SaveIcon />}>
                    Sauvegarder le Portefeuille
                </Button>
                <Button onClick={() => setOpenScenario(true)} variant="contained" startIcon={<SimulationIcon />}>Simuler un Scénario</Button>
                <Button onClick={generateReport} variant="contained" startIcon={<ReportIcon />}>Générer un Rapport</Button>
            </Box>

            <Dialog
                open={openScenario}
                onClose={() => setOpenScenario(false)}
                fullWidth
                maxWidth="md"
            >
                <DialogTitle>Simulation de Scénario</DialogTitle>
                <DialogContent>
                    <Select
                        value={selectedScenario}
                        onChange={(e) => setSelectedScenario(e.target.value)}
                        fullWidth
                        sx={{ mb: 2 }}
                    >
                        <MenuItem value="market_crash">Krach Boursier</MenuItem>
                        <MenuItem value="bull_market">Marché Haussier</MenuItem>
                        <MenuItem value="high_inflation">Forte Inflation</MenuItem>
                    </Select>
                    <Button onClick={simulateScenario} variant="contained" sx={{ mb: 2 }}>Lancer la Simulation</Button>
                    {scenarioResults && scenarioResults.daily_returns && scenarioResults.daily_returns.length > 0 ? (
                        <Box>
                            <Typography>Scénario: {scenarioResults.scenario}</Typography>
                            <Typography>Valeur Initiale: {scenarioResults.initial_value.toFixed(2)} €</Typography>
                            <Typography>Valeur Finale: {scenarioResults.final_value.toFixed(2)} €</Typography>
                            <Typography>Rendement Total: {(scenarioResults.total_return * 100).toFixed(2)}%</Typography>
                            <Box sx={{ width: '100%', height: 400 }}>
                                <ResponsiveContainer width="100%" height={400}>
                                    <LineChart data={scenarioResults.portfolio_values.map((value, index) => ({ day: index, value: value }))}>
                                        <XAxis
                                            dataKey="day"
                                            tickFormatter={(tick) => `Jour ${tick + 1}`}
                                        />
                                        <YAxis
                                            tickFormatter={(value) => `$${value.toFixed(0)}`}
                                        />
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <RechartsTooltip
                                            formatter={(value) => [`$${value.toFixed(2)}`, "Valeur du Portefeuille"]}
                                            labelFormatter={(label) => `Jour ${label + 1}`}
                                        />
                                        <Legend />
                                        <Line type="monotone" dataKey="value" stroke="#8884d8" dot={false} name="Valeur du Portefeuille" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </Box>
                        </Box>
                    ) : (
                        <Typography>Aucune donnée de scénario disponible</Typography>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOpenScenario(false)}>Fermer</Button>
                </DialogActions>
            </Dialog>

            <Dialog open={openReport} onClose={() => setOpenReport(false)} maxWidth="md" fullWidth>
                <DialogTitle>Rapport du Portefeuille</DialogTitle>
                <DialogContent>
                    {reportData ? (
                        <iframe
                            src={`data:application/pdf;base64,${reportData}`}
                            width="100%"
                            height="500px"
                            style={{ border: 'none' }}
                        />
                    ) : (
                        <Typography>Aucune donnée de rapport disponible</Typography>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOpenReport(false)}>Fermer</Button>
                    {reportData && (
                        <Button onClick={() => {
                            const linkSource = `data:application/pdf;base64,${reportData}`;
                            const downloadLink = document.createElement("a");
                            downloadLink.href = linkSource;
                            downloadLink.download = "rapport_portefeuille.pdf";
                            downloadLink.click();
                        }}>
                            Télécharger le PDF
                        </Button>
                    )}
                </DialogActions>
            </Dialog>
        </Paper>
    );
}

export default Portfolio;