import React, { useState, useEffect, useCallback } from 'react';
import {
    TextField, Button, Table, TableBody, TableCell, TableHead, TableRow, Paper, List, ListItem, ListItemText, Typography, Dialog, DialogTitle, DialogContent, DialogActions, Select, MenuItem, CircularProgress, InputAdornment, Autocomplete, IconButton, Box, Switch, FormControlLabel, Tooltip, LinearProgress
} from '@mui/material';
import {
    PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip as RechartsTooltip, LineChart, Line, XAxis, YAxis, CartesianGrid
} from 'recharts';
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

    const COLORS = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];

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
    const [reportProgress, setReportProgress] = useState(0);
    const [currentStep, setCurrentStep] = useState('');
    const [generatingReport, setGeneratingReport] = useState(false);

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

    useEffect(() => {
        if (portfolio.stocks && portfolio.stocks.length > 0) {
            fetchLatestPrices();
        }
    }, [portfolio]);

    useEffect(() => {
        const fetchNews = async () => {
            if (!portfolio || !portfolio.stocks) {
                console.error("Portfolio or portfolio.stocks is undefined");
                return;
            }
            const tickers = portfolio.stocks.map(stock => stock.symbol).join(',');
            try {
                const response = await axios.get(`${process.env.REACT_APP_API_URL}/news?tickers=${tickers}`);
                setNews(response.data);
            } catch (error) {
                console.error("Erreur lors de la récupération des nouvelles:", error);
            }
        };
        fetchNews();
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

    const fetchLatestPrices = async () => {
        if (!portfolio || !portfolio.stocks) {
            console.error("Portfolio or portfolio.stocks is undefined");
            return;
        }
        const updatedPrices = { ...livePrices };
        for (const stock of portfolio.stocks) {
            try {
                const response = await axios.get(`${process.env.REACT_APP_API_URL}/latest_price?symbol=${stock.symbol}`);
                if (response.data && response.data.price) {
                    updatedPrices[stock.symbol] = response.data.price;
                }
            } catch (error) {
                console.error(`Erreur lors de la récupération du prix pour ${stock.symbol}:`, error);
            }
        }
        setLivePrices(updatedPrices);
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
            console.error("Error simulating scenario:", error);
            alert('Error simulating scenario. Please try again.');
        }
    };

    const generateReport = async () => {
        try {
            setGeneratingReport(true);
            setReportProgress(0);
            setCurrentStep('Initialisation de la génération du rapport');

            const eventSource = new EventSource(`${process.env.REACT_APP_API_URL}/generate_report`, {
                withCredentials: true
            });

            eventSource.onerror = (error) => {
                console.error("La connexion EventSource a échoué:", error);
                eventSource.close();
                setGeneratingReport(false);
                alert('Erreur lors de la génération du rapport. Veuillez réessayer.');
            };

            eventSource.onopen = () => {
                console.log("Connexion EventSource établie");
            };

            eventSource.onmessage = (event) => {
                console.log("Message reçu:", event.data);
                const data = JSON.parse(event.data.replace("'", '"')); // Remplacer les apostrophes si nécessaire
                setReportProgress(data.progress);
                setCurrentStep(data.step);

                if (data.report) {
                    setReportData(data.report);
                    setOpenReport(true);
                    eventSource.close();
                    setGeneratingReport(false);
                }
            };

        } catch (error) {
            console.error("Erreur dans generateReport:", error);
            setGeneratingReport(false);
            alert('Erreur lors de la génération du rapport. Veuillez réessayer.');
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

    const calculateDiff = (entryPrice, currentPrice, shares) => {
        const diff = (currentPrice - entryPrice) * shares;
        const color = diff >= 0 ? 'green' : 'red';
        return { diff, color };
    };

    if (loading) {
        return <CircularProgress />;
    }

    return (
        <Paper sx={{ padding: 2 }}>
            <Typography variant="h6">Latest News</Typography>
            {news && news.length > 0 ? (
                <List>
                    {news.map((item, index) => (
                        <ListItem key={index}>
                            <ListItemText primary={item.title} secondary={item.description} />
                        </ListItem>
                    ))}
                </List>
            ) : (
                <Typography>No news available for the current stocks</Typography>
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
                                <TableCell>Symbol</TableCell>
                                <TableCell>{displayMode === 'weight' ? 'Weight (%)' : 'Shares'}</TableCell>
                                <TableCell>Entry Price</TableCell>
                                <TableCell>Current Price</TableCell>
                                <TableCell>Value</TableCell>
                                <TableCell>Diff</TableCell>
                                <TableCell>Actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {portfolio.stocks.map((stock, index) => {
                                const shares = displayMode === 'weight'
                                    ? (parseFloat(stock.weight) / 100) * portfolioValue / parseFloat(stock.entry_price)
                                    : parseFloat(stock.weight);
                                const currentPrice = livePrices[stock.symbol] || parseFloat(stock.entry_price);
                                const value = shares * currentPrice;
                                const { diff, color } = calculateDiff(parseFloat(stock.entry_price), currentPrice, shares);
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
                                        <TableCell>{currentPrice ? formatNumber(currentPrice) : 'Loading...'}</TableCell>
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
                    <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>Portfolio Allocation</Typography>
                    <PortfolioPieChart portfolio={portfolio} />
                </>
            ) : (
                <Typography sx={{ my: 2 }}>Le portefeuille est actuellement vide. Ajoutez des actions pour commencer.</Typography>
            )}

            <Box sx={{ mt: 3 }}>
                <TextField
                    label="Portfolio Value"
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
                <Typography variant="h6" sx={{ mt: 2 }}>Current Portfolio Value: {portfolioValue.toFixed(2)}</Typography>
                <FormControlLabel
                    control={<Switch checked={displayMode === 'shares'} onChange={() => setDisplayMode(displayMode === 'weight' ? 'shares' : 'weight')} />}
                    label={`Switch to ${displayMode === 'weight' ? 'Shares' : 'Weight'} Mode`}
                    sx={{ mt: 2 }}
                />
            </Box>

            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button onClick={() => savePortfolio(portfolio.stocks)} variant="contained" color="secondary">
                    Save Portfolio
                </Button>
                <Button onClick={() => setOpenScenario(true)} variant="contained">Simulate Scenario</Button>
                <Button onClick={generateReport} variant="contained">Generate Report</Button>
            </Box>

            <Dialog
                open={openScenario}
                onClose={() => setOpenScenario(false)}
                fullWidth
                maxWidth="md"
            >
                <DialogTitle>Scenario Simulation</DialogTitle>
                <DialogContent>
                    <Select
                        value={selectedScenario}
                        onChange={(e) => setSelectedScenario(e.target.value)}
                        fullWidth
                        sx={{ mb: 2 }}
                    >
                        <MenuItem value="market_crash">Market Crash</MenuItem>
                        <MenuItem value="bull_market">Bull Market</MenuItem>
                        <MenuItem value="high_inflation">High Inflation</MenuItem>
                    </Select>
                    <Button onClick={simulateScenario} variant="contained" sx={{ mb: 2 }}>Run Simulation</Button>
                    {scenarioResults && scenarioResults.daily_returns && scenarioResults.daily_returns.length > 0 ? (
                        <Box>
                            <Typography>Scenario: {scenarioResults.scenario}</Typography>
                            <Typography>Initial Value: ${scenarioResults.initial_value.toFixed(2)}</Typography>
                            <Typography>Final Value: ${scenarioResults.final_value.toFixed(2)}</Typography>
                            <Typography>Total Return: {(scenarioResults.total_return * 100).toFixed(2)}%</Typography>
                            <Box sx={{ width: '100%', height: 400 }}>
                            <ResponsiveContainer width="100%" height={400}>
                                <LineChart data={scenarioResults.portfolio_values.map((value, index) => ({ day: index, value: value }))}>
                                    <XAxis 
                                        dataKey="day"
                                        tickFormatter={(tick) => `Day ${tick + 1}`}
                                    />
                                    <YAxis 
                                        tickFormatter={(value) => `$${value.toFixed(0)}`}
                                    />
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <Tooltip 
                                        formatter={(value) => [`$${value.toFixed(2)}`, "Portfolio Value"]}
                                        labelFormatter={(label) => `Day ${label + 1}`}
                                    />
                                    <Legend />
                                    <Line type="monotone" dataKey="value" stroke="#8884d8" dot={false} name="Portfolio Value" />
                                </LineChart>
                            </ResponsiveContainer>
                            </Box>
                        </Box>
                    ) : (
                        <Typography>No scenario data available</Typography>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOpenScenario(false)}>Close</Button>
                </DialogActions>
            </Dialog>
            {generatingReport && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 2 }}>
                    <CircularProgress variant="determinate" value={reportProgress} />
                    <Typography variant="caption" sx={{ mt: 1 }}>{`${Math.round(reportProgress)}%`}</Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>{currentStep}</Typography>
                </Box>
            )}

            {reportData && (
                <Dialog open={openReport} onClose={() => setOpenReport(false)} maxWidth="md" fullWidth>
                    <DialogTitle>Portfolio Report</DialogTitle>
                    <DialogContent>
                        <iframe
                            src={`data:application/pdf;base64,${reportData}`}
                            width="100%"
                            height="500px"
                            style={{ border: 'none' }}
                        />
                    </DialogContent>
                    <DialogActions>
                        <Button onClick={() => setOpenReport(false)}>Close</Button>
                        <Button onClick={() => {
                            const linkSource = `data:application/pdf;base64,${reportData}`;
                            const downloadLink = document.createElement("a");
                            downloadLink.href = linkSource;
                            downloadLink.download = "portfolio_report.pdf";
                            downloadLink.click();
                        }}>
                            Download PDF
                        </Button>
                    </DialogActions>
                </Dialog>
            )}
        </Paper>
    );
}

export default Portfolio;